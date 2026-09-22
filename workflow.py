#!/usr/bin/env python3
"""YAML-driven split and training entry points; paths are relative to this repository."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a nonempty path string in YAML")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_yaml(path):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return config


def run_split(config_path, *, dry_run=False):
    recipe = read_yaml(config_path)
    allowed = {"data_dir", "output", "expected_seeds", "split_seed", "real", "synthetic"}
    if set(recipe) - allowed:
        raise ValueError(f"Unknown split settings: {set(recipe) - allowed}")
    data_dir = project_path(recipe["data_dir"])
    outputs = recipe["output"]
    if set(outputs) != {"data_config", "manifest"}:
        raise ValueError("output must contain data_config and manifest")
    data_config = project_path(outputs["data_config"])
    manifest_path = project_path(outputs["manifest"])
    if len({Path(config_path).resolve(), data_config, manifest_path}) != 3:
        raise ValueError("Split recipe, data config and manifest must be different files")
    if data_config.exists():
        existing = read_yaml(data_config)
        if set(existing) != {"data", "split", "split_manifest"}:
            raise ValueError(f"Refusing to overwrite a file that is not a generated data config: {data_config}")
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if (not isinstance(existing_manifest, dict)
                or set(existing_manifest.get("partitions", {})) != {"train", "validation", "test"}
                or existing_manifest.get("strategy") != "track_family_holdout"):
            raise ValueError(f"Refusing to overwrite a file that is not a split manifest: {manifest_path}")
    if not (data_dir / "dataset_manifest.yaml").is_file():
        raise FileNotFoundError(f"Generate the dataset first: {data_dir / 'dataset_manifest.yaml'}")
    seeds = recipe["expected_seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("expected_seeds must be unique nonnegative integers")
    if type(recipe.get("split_seed", 42)) is not int or recipe.get("split_seed", 42) < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    families = []
    for source in ("real", "synthetic"):
        for partition in ("train", "validation", "test"):
            groups = recipe[source][partition]
            if not isinstance(groups, list) or any(not isinstance(group, str) or not group for group in groups):
                raise ValueError(f"{source}.{partition} must be a list of family names")
            families.extend(groups)
    if len(families) != len(set(families)):
        raise ValueError("Each track family must occur in exactly one partition")
    if recipe["synthetic"]["test"]:
        raise ValueError("The final test partition must contain real tracks only")
    print(f"Data: {data_dir}\nData config: {data_config}\nManifest: {manifest_path}", flush=True)
    if dry_run:
        print(yaml.safe_dump(recipe, sort_keys=False))
        return
    from prepare_mixed_split import prepare
    return prepare(data_dir, config_path, data_config, manifest_path, data_only=True)


def resolve_training(config_path):
    recipe = read_yaml(config_path)
    required = {"data_config", "data", "model", "training", "augmentation", "save", "runtime"}
    if set(recipe) != required:
        raise ValueError(f"Training YAML needs exactly these sections: {sorted(required)}")
    data_config = project_path(recipe["data_config"])
    prepared = read_yaml(data_config)
    if set(prepared) != {"data", "split", "split_manifest"}:
        raise ValueError("Run split_data.sh to produce a data-only config")
    if set(recipe["data"]) - {"input_mode", "num_classes", "normalize"}:
        raise ValueError("Training YAML may only change data.input_mode, num_classes and normalize; change partitions in split_data.yaml")
    config = {key: copy.deepcopy(recipe[key]) for key in ("data", "model", "training", "augmentation", "save")}
    config["data"] = {**prepared["data"], **config["data"]}
    config["split"] = copy.deepcopy(prepared["split"])

    # Ensure the generated config still agrees with its three-way split manifest.
    manifest_path = project_path(prepared["split_manifest"])
    manifest = json.loads(manifest_path.read_text())
    parts = manifest["partitions"]
    expected_tracks = sorted(parts["train"]["tracks"] + parts["validation"]["tracks"])
    if (
        sorted(config["data"]["include_dirs"]) != expected_tracks
        or config["data"]["exclude_dirs"]
        or set(expected_tracks) & set(parts["test"]["tracks"])
        or config["split"]["strategy"] != "track_holdout"
        or sorted(config["split"]["val_groups"]) != sorted(parts["validation"]["groups"])
        or config["data"].get("expected_sha256") != manifest["train_validation_sha256"]
        or project_path(config["data"]["data_dir"]) != project_path(manifest["data_dir"])
        or config["data"]["coordinate_frame"] != manifest["coordinate_frame"]
        or config["data"]["side_semantics"] != manifest["side_semantics"]
        or config["data"]["allow_legacy_without_metadata"]
    ):
        raise ValueError("Prepared data config disagrees with the split manifest; rerun split_data.sh")
    for key in ("epochs", "batch_size"):
        value = config["training"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    runtime = recipe["runtime"]
    if set(runtime) != {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"}:
        raise ValueError("runtime must set OMP_NUM_THREADS, MKL_NUM_THREADS and CUBLAS_WORKSPACE_CONFIG")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError(f"runtime.{key} must be a positive integer")
    if runtime["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' or ':16:8'")
    config["data"]["data_dir"] = str(project_path(config["data"]["data_dir"]))
    config["save"]["save_dir"] = str(project_path(config["save"]["save_dir"]))
    checkpoint = config["save"]["checkpoint_name"]
    if not isinstance(checkpoint, str) or Path(checkpoint).name != checkpoint or checkpoint in {"", ".", ".."}:
        raise ValueError("save.checkpoint_name must be a filename, not a path")
    return recipe, config, data_config, manifest_path


def run_training(config_path, *, dry_run=False):
    recipe, config, data_config, manifest_path = resolve_training(config_path)
    save_dir = Path(config["save"]["save_dir"])
    if save_dir.exists() and (not save_dir.is_dir() or any(save_dir.iterdir())):
        raise FileExistsError(f"Choose a new save.save_dir; output already exists: {save_dir}")
    print(yaml.safe_dump({"runtime": recipe["runtime"], **config}, sort_keys=False), flush=True)
    if dry_run:
        return config
    save_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = save_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (save_dir / "run_settings.yaml").write_text(Path(config_path).read_text())
    (save_dir / "data_config.yaml").write_text(data_config.read_text())
    (save_dir / "dataset_split.json").write_text(manifest_path.read_text())
    env = dict(os.environ, **{key: str(value) for key, value in recipe["runtime"].items()})
    # Apply CUDA/BLAS settings before the child process imports torch.
    subprocess.run(
        [sys.executable, "-u", str(PROJECT_ROOT / "train_sidenet.py"), str(resolved_path)],
        cwd=PROJECT_ROOT, env=env, check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("split", "train"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings without writing outputs or running training")
    args = parser.parse_args()
    try:
        path = project_path(args.config)
        if args.stage == "split":
            run_split(path, dry_run=args.dry_run)
        else:
            run_training(path, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
