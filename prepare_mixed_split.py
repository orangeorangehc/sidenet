#!/usr/bin/env python3
"""Audit a mixed multi-seed corpus and save a family split and training config."""

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from sidenet_data import canonical_track_group, dataset_fingerprint, load_frames, split_frames


PROJECT_ROOT = Path(__file__).resolve().parent


def summarize(samples):
    left = sum(int((s.labels == 0).sum()) for s in samples)
    right = sum(int((s.labels == 1).sum()) for s in samples)
    return {
        "num_frames": len(samples),
        "num_cones": left + right,
        "label_counts": {"Left": left, "Right": right},
        "groups": sorted({canonical_track_group(s.track) for s in samples}),
        "tracks": sorted({s.track for s in samples}),
    }


def prepare(data_dir, policy_path, config_output, manifest_output, *, data_only=False):
    policy = yaml.safe_load(Path(policy_path).read_text())
    expected_seeds = set(policy["expected_seeds"])
    membership = {}
    for source_type in ("real", "synthetic"):
        for partition in ("train", "validation", "test"):
            for group in policy[source_type][partition]:
                if group in membership:
                    raise ValueError(f"Family assigned more than once: {group}")
                membership[group] = (source_type, partition)
    if policy["synthetic"]["test"]:
        raise ValueError("This recipe reserves the final test for real track maps")

    samples, summary = load_frames(
        data_dir, expected_coordinate_frame="ego", expected_side_semantics="track_global"
    )
    observed_groups = {canonical_track_group(s.track) for s in samples}
    if observed_groups != set(membership):
        raise ValueError(f"Policy/dataset family mismatch: {observed_groups ^ set(membership)}")
    sources = {}
    source_seeds = defaultdict(set)
    for name in summary["tracks"]:
        meta = yaml.safe_load((Path(data_dir) / name / "metadata.yaml").read_text())
        source = meta["source"]
        group = canonical_track_group(name)
        if source["type"] != membership[group][0]:
            raise ValueError(f"Source type mismatch for {name}")
        if name != f"{source['track']}_s{source['seed']}":
            raise ValueError(f"Name/source mismatch for {name}")
        if meta["config"]["seed"] != source["seed"]:
            raise ValueError(f"Seed mismatch for {name}")
        sources[name] = source
        source_seeds[(source["type"], source["track"])].add(source["seed"])
    if not source_seeds or any(seeds != expected_seeds for seeds in source_seeds.values()):
        raise ValueError("Every source track must contain exactly the configured seeds")

    partitions = {key: [] for key in ("train", "validation", "test")}
    for sample in samples:
        if not torch.isfinite(sample.points).all() or not set(sample.labels.tolist()) <= {0, 1}:
            raise ValueError(f"Invalid coordinates or labels: {sample.path}")
        rows = [line.split() for line in sample.path.read_text().splitlines() if line.strip()]
        if len(rows) != len(sample.labels) or any(
            len(row) != 9 or row[-1] not in {"Cone_Left", "Cone_Right"} for row in rows
        ):
            raise ValueError(f"Invalid training row: {sample.path}")
        partition = membership[canonical_track_group(sample.track)][1]
        partitions[partition].append(sample)

    result = {}
    used_ids, used_groups = set(), set()
    for name, part in partitions.items():
        stats = summarize(part)
        ids = {s.sample_id for s in part}
        groups = set(stats["groups"])
        if not part or len(ids) != len(part) or ids & used_ids or groups & used_groups:
            raise ValueError(f"Empty, duplicated or leaking partition: {name}")
        if min(stats["label_counts"].values()) == 0:
            raise ValueError(f"Partition lacks a label class: {name}")
        used_ids.update(ids)
        used_groups.update(groups)
        result[name] = {
            **stats,
            "by_source": {
                origin: summarize([s for s in part if sources[s.track]["type"] == origin])
                for origin in ("real", "synthetic")
            },
            "frame_ids": sorted(ids),
            "dataset_sha256": dataset_fingerprint(part),
        }

    config = copy.deepcopy(yaml.safe_load((PROJECT_ROOT / "configs/dgcnn.yaml").read_text()))
    config["data"]["data_dir"] = os.path.relpath(Path(data_dir).resolve(), PROJECT_ROOT)
    config["data"]["include_dirs"] = sorted(
        set(result["train"]["tracks"] + result["validation"]["tracks"])
    )
    config["data"]["exclude_dirs"] = []
    config["split"]["val_groups"] = result["validation"]["groups"]
    config["split"]["seed"] = policy.get("split_seed", config["split"]["seed"])
    config["save"]["save_dir"] = "runs/dgcnn_mixed"
    trainval, _ = load_frames(
        data_dir, include_dirs=config["data"]["include_dirs"],
        expected_coordinate_frame="ego", expected_side_semantics="track_global",
    )
    train, val, actual = split_frames(trainval, config["split"])
    if (
        {s.sample_id for s in train} != set(result["train"]["frame_ids"])
        or {s.sample_id for s in val} != set(result["validation"]["frame_ids"])
    ):
        raise ValueError("Training configuration does not reproduce the intended split")

    manifest = {
        "schema_version": 1,
        "data_dir": config["data"]["data_dir"],
        "strategy": "track_family_holdout",
        "seed": config["split"]["seed"],
        "observation_seeds": sorted(expected_seeds),
        "coordinate_frame": "ego",
        "side_semantics": "track_global",
        "dataset_sha256": dataset_fingerprint(samples),
        "train_validation_sha256": actual["dataset_sha256"],
        "num_frames": len(samples),
        "num_cones": summary["num_cones"],
        "group_overlap": [],
        "policy": policy,
        "partitions": result,
    }
    for path in (config_output, manifest_output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if data_only:
        # Keep user-edited model/training settings outside the generated split.
        data = {key: value for key, value in config["data"].items()
                if key not in {"input_mode", "num_classes", "normalize"}}
        data["expected_sha256"] = actual["dataset_sha256"]
        config = {
            "data": data,
            "split": {key: value for key, value in config["split"].items() if key != "val_ratio"},
            "split_manifest": os.path.relpath(Path(manifest_output).resolve(), PROJECT_ROOT),
        }
    Path(config_output).write_text(
        "# Generated by prepare_mixed_split.py; final test tracks are excluded.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    Path(manifest_output).write_text(json.dumps(manifest, indent=2) + "\n")
    for name, info in result.items():
        print(f"{name}: {info['num_frames']} frames, {info['num_cones']} cones, groups={info['groups']}")
        for origin, stats in info["by_source"].items():
            print(f"  {origin}: {stats['num_frames']} frames, {stats['num_cones']} cones")
    print(f"Saved {config_output} and {manifest_output}; family overlap = 0")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT.parent / "bitfsd-generator/output/sidenet_data_mixed_ego")
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "configs/mixed_split.yaml")
    parser.add_argument("--config-output", type=Path, default=PROJECT_ROOT / "configs/dgcnn_mixed.yaml")
    parser.add_argument("--manifest-output", type=Path, default=PROJECT_ROOT / "splits/mixed_ego.json")
    args = parser.parse_args()
    prepare(args.data_dir, args.policy, args.config_output, args.manifest_output)
