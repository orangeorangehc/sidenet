#!/usr/bin/env python3
"""Train and evaluate SideNet with reproducible, track-aware splits."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sidenet import build_model
from sidenet_data import (
    FrameSample,
    augment_frame,
    load_frames,
    normalize_frame,
    select_input_features,
    set_global_seed,
    split_frames,
    write_split_manifest,
)


CLI_CONFIG_KEYS = {
    "epochs": "training.epochs",
    "lr": "training.lr",
    "batch_size": "training.batch_size",
    "seed": "training.seed",
    "split_seed": "split.seed",
    "device": "training.device",
    "save_dir": "save.save_dir",
    "data_dir": "data.data_dir",
    "split_strategy": "split.strategy",
    "val_groups": "split.val_groups",
    "allow_legacy_data": "data.allow_legacy_without_metadata",
}


def _set_nested(config: dict, dotted_key: str, value) -> None:
    section, _, key = dotted_key.partition(".")
    if not key:
        config[section] = value
        return
    if section not in config or not isinstance(config[section], dict):
        config[section] = {}
    config[section][key] = value


def load_config(
    config_path: str | Path | None = None, cli_overrides: dict | None = None
) -> dict:
    """Load YAML and apply CLI values to their real nested config keys."""
    path = (
        Path(config_path)
        if config_path
        else Path(__file__).resolve().parent / "config.yaml"
    )
    with path.resolve().open() as handle:
        config = yaml.safe_load(handle) or {}
    for key, value in (cli_overrides or {}).items():
        if value is not None:
            _set_nested(config, CLI_CONFIG_KEYS.get(key, key), value)
    return config


def resolve_config_path(config_path: str | Path | None) -> Path:
    return (
        Path(config_path).resolve()
        if config_path
        else (Path(__file__).resolve().parent / "config.yaml").resolve()
    )


def resolve_runtime_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    # Repository configs may live in ``configs/``; keep their runtime paths
    # consistently relative to the SideNet project root, not the YAML folder.
    project_root = Path(__file__).resolve().parent
    return path if path.is_absolute() else (project_root / path).resolve()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_labels(samples: list[FrameSample], num_classes: int) -> None:
    for sample in samples:
        if sample.labels is None:
            raise ValueError(f"Training sample has no labels: {sample.path}")
        if sample.labels.numel() and int(sample.labels.max()) >= num_classes:
            raise ValueError(
                f"{sample.path} contains class id {int(sample.labels.max())}, "
                f"but data.num_classes={num_classes}"
            )


def _prepare_sample(
    sample: FrameSample,
    *,
    device: torch.device,
    input_mode: str,
    normalize: bool,
    augmentation: dict | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = sample.points.to(device)
    labels = sample.labels.to(device)
    if normalize:
        points = normalize_frame(points)
    if augmentation and augmentation.get("enabled", True):
        points, labels = augment_frame(
            points,
            labels,
            rot_range=float(augmentation.get("rot_range", 0.2618)),
            scale_range=augmentation.get("scale_range", [0.97, 1.03]),
            noise_std=float(augmentation.get("noise_std", 0.02)),
            flip_prob=float(augmentation.get("flip_prob", 0.5)),
            transform_origin=augmentation.get("transform_origin", "ego"),
        )
    return select_input_features(points, input_mode), labels


def train_epoch(
    model,
    optimizer,
    samples: list[FrameSample],
    batch_size: int,
    augmentation: dict,
    *,
    input_mode: str,
    normalize: bool,
    device: torch.device,
) -> tuple[float, float]:
    """Train without cross-frame padding; one optimizer step still spans a frame batch."""
    model.train()
    indices = torch.randperm(len(samples)).tolist()
    loss_sum = 0.0
    correct = 0
    total = 0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        batch_loss = None
        batch_cones = 0

        for index in batch_indices:
            points, labels = _prepare_sample(
                samples[index],
                device=device,
                input_mode=input_mode,
                normalize=normalize,
                augmentation=augmentation,
            )
            logits = model(points)
            frame_loss = F.cross_entropy(logits, labels, reduction="sum")
            batch_loss = frame_loss if batch_loss is None else batch_loss + frame_loss
            batch_cones += labels.numel()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

        if batch_loss is None or batch_cones == 0:
            continue
        (batch_loss / batch_cones).backward()
        optimizer.step()
        loss_sum += batch_loss.detach().item()

    if total == 0:
        raise RuntimeError("No labeled cones were available for training")
    return loss_sum / total, correct / total


def _metrics_from_counts(
    correct: int,
    total: int,
    frame_accuracies: list[float],
    track_counts: dict[str, list[int]],
    confusion: torch.Tensor,
) -> dict:
    per_track = {
        track: values[0] / values[1]
        for track, values in sorted(track_counts.items())
        if values[1] > 0
    }
    confusion_float = confusion.float()
    true_support = confusion_float.sum(dim=1)
    predicted_support = confusion_float.sum(dim=0)
    true_positive = confusion_float.diag()
    recall = true_positive / true_support.clamp(min=1.0)
    precision = true_positive / predicted_support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = true_support > 0
    return {
        "accuracy": correct / total,
        "balanced_accuracy": recall[present].mean().item(),
        "macro_f1": f1[present].mean().item(),
        "per_class_recall": recall.tolist(),
        "macro_frame_accuracy": sum(frame_accuracies) / len(frame_accuracies),
        "macro_track_accuracy": sum(per_track.values()) / len(per_track),
        "worst_track_accuracy": min(per_track.values()),
        "per_track_accuracy": per_track,
        "num_frames": len(frame_accuracies),
        "num_cones": total,
        "confusion": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(
    model,
    samples: list[FrameSample],
    *,
    input_mode: str,
    normalize: bool,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    correct = 0
    total = 0
    frame_accuracies = []
    track_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    nll_sum = 0.0
    calibration_count = torch.zeros(15, dtype=torch.long)
    calibration_confidence = torch.zeros(15, dtype=torch.float64)
    calibration_correct = torch.zeros(15, dtype=torch.float64)

    for sample in samples:
        points, labels = _prepare_sample(
            sample,
            device=device,
            input_mode=input_mode,
            normalize=normalize,
            augmentation=None,
        )
        logits = model(points)
        probabilities = logits.softmax(dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        nll_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        frame_correct = (predictions == labels).sum().item()
        frame_total = labels.numel()
        correct += frame_correct
        total += frame_total
        frame_accuracies.append(frame_correct / frame_total)
        track_counts[sample.track][0] += frame_correct
        track_counts[sample.track][1] += frame_total
        flat = labels.cpu() * num_classes + predictions.cpu()
        confusion += torch.bincount(flat, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
        bins = (confidence.detach().cpu() * 15).long().clamp(max=14)
        calibration_count += torch.bincount(bins, minlength=15)
        calibration_confidence += torch.bincount(
            bins, weights=confidence.detach().cpu().double(), minlength=15
        )
        calibration_correct += torch.bincount(
            bins,
            weights=(predictions == labels).detach().cpu().double(),
            minlength=15,
        )

    if total == 0:
        raise RuntimeError("No labeled cones were available for evaluation")
    metrics = _metrics_from_counts(
        correct, total, frame_accuracies, track_counts, confusion
    )
    populated = calibration_count > 0
    bin_accuracy = calibration_correct[populated] / calibration_count[populated]
    bin_confidence = calibration_confidence[populated] / calibration_count[populated]
    metrics["nll"] = nll_sum / total
    metrics["ece_15_bin"] = (
        ((bin_accuracy - bin_confidence).abs() * calibration_count[populated]).sum()
        / total
    ).item()
    return metrics


@torch.no_grad()
def evaluate_baseline(
    samples: list[FrameSample], predictor, num_classes: int = 2
) -> dict:
    correct = 0
    total = 0
    frame_accuracies = []
    track_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for sample in samples:
        labels = sample.labels
        predictions = predictor(sample)
        frame_correct = (predictions == labels).sum().item()
        frame_total = labels.numel()
        correct += frame_correct
        total += frame_total
        frame_accuracies.append(frame_correct / frame_total)
        track_counts[sample.track][0] += frame_correct
        track_counts[sample.track][1] += frame_total
        flat = labels * num_classes + predictions
        confusion += torch.bincount(flat, minlength=num_classes**2).reshape(
            num_classes, num_classes
        )
    return _metrics_from_counts(
        correct, total, frame_accuracies, track_counts, confusion
    )


def run_knn_baseline(
    train_samples: list[FrameSample], val_samples: list[FrameSample], model_cfg: dict
) -> dict:
    """A deployable train-reference k-NN, not the former GT-neighbor oracle."""
    reference_points = torch.cat(
        [sample.points[:, :2] for sample in train_samples], dim=0
    )
    reference_labels = torch.cat([sample.labels for sample in train_samples], dim=0)
    k = int(model_cfg.get("knn_k", 5))
    weighted = bool(model_cfg.get("knn_weighted", False))

    def predict(sample: FrameSample) -> torch.Tensor:
        distances = torch.cdist(sample.points[:, :2], reference_points)
        neighbor_distances, indices = distances.topk(
            min(k, reference_points.shape[0]), dim=-1, largest=False
        )
        neighbor_labels = reference_labels[indices]
        if not weighted:
            return neighbor_labels.mode(dim=-1)[0]
        weights = 1.0 / neighbor_distances.clamp(min=1e-8)
        right_probability = (neighbor_labels.float() * weights).sum(
            dim=-1
        ) / weights.sum(dim=-1)
        return (right_probability >= 0.5).long()

    return evaluate_baseline(val_samples, predict)


def run_lateral_baseline(val_samples: list[FrameSample], coordinate_frame: str) -> dict:
    if coordinate_frame != "ego":
        raise ValueError("The lateral sign baseline is only meaningful in ego frame")

    def predict(sample: FrameSample) -> torch.Tensor:
        # Standard LiDAR/ego convention: +x forward, +y left.
        return torch.where(
            sample.points[:, 1] >= 0,
            torch.zeros(sample.points.shape[0], dtype=torch.long),
            torch.ones(sample.points.shape[0], dtype=torch.long),
        )

    return evaluate_baseline(val_samples, predict)


def _state_dict_cpu(model) -> dict:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def runtime_provenance() -> dict:
    project_root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--short"], cwd=project_root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "sidenet_git_commit": commit,
        "sidenet_worktree_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def _print_metrics(prefix: str, metrics: dict) -> None:
    print(
        f"{prefix}: micro={metrics['accuracy']:.4f} "
        f"balanced={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} "
        f"macro_frame={metrics['macro_frame_accuracy']:.4f} "
        f"macro_track={metrics['macro_track_accuracy']:.4f} "
        f"worst_track={metrics['worst_track_accuracy']:.4f}"
    )
    for track, accuracy in metrics["per_track_accuracy"].items():
        print(f"  {track}: {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, nargs="?", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--save-dir", "--save_dir", dest="save_dir", type=str)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", type=str)
    parser.add_argument("--split-strategy", choices=["random_frame", "track_holdout"])
    parser.add_argument("--val-groups", nargs="+")
    parser.add_argument(
        "--allow-legacy-data",
        action="store_true",
        default=None,
        help="Allow datasets without metadata.yaml (explicit compatibility mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        key: getattr(args, key)
        for key in CLI_CONFIG_KEYS
        if hasattr(args, key) and getattr(args, key) is not None
    }
    config_path = resolve_config_path(args.config)
    config = load_config(config_path, overrides)
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    split_cfg = config.get(
        "split", {"strategy": "track_holdout", "seed": 42, "val_ratio": 0.2}
    )
    augmentation_cfg = config.get("augmentation", {"enabled": True})
    save_cfg = config.get("save", {})

    model_seed = int(training_cfg.get("seed", 42))
    split_seed = int(split_cfg.get("seed", 42))
    set_global_seed(
        model_seed, deterministic=bool(training_cfg.get("deterministic", True))
    )
    device = resolve_device(training_cfg.get("device", "auto"))
    data_dir = resolve_runtime_path(data_cfg["data_dir"], config_path)
    save_dir = resolve_runtime_path(save_cfg.get("save_dir", "runs"), config_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    samples, data_summary = load_frames(
        data_dir,
        include_dirs=data_cfg.get("include_dirs"),
        exclude_dirs=data_cfg.get("exclude_dirs"),
        expected_coordinate_frame=data_cfg.get("coordinate_frame"),
        expected_side_semantics=data_cfg.get("side_semantics"),
        allow_legacy_without_metadata=bool(
            data_cfg.get("allow_legacy_without_metadata", False)
        ),
        assumed_coordinate_frame=data_cfg.get("assumed_legacy_coordinate_frame"),
        require_labels=True,
    )
    num_classes = int(data_cfg.get("num_classes", 2))
    _validate_labels(samples, num_classes)
    train_samples, val_samples, split_manifest = split_frames(samples, split_cfg)
    write_split_manifest(split_manifest, save_dir / "split_manifest.json")

    print("Configuration:")
    print(f"  config: {config_path}")
    print(f"  data: {data_dir}")
    print(
        f"  contract: frame={data_cfg.get('coordinate_frame')}, "
        f"side={data_cfg.get('side_semantics')}, input={data_cfg.get('input_mode')}"
    )
    print(
        f"  split: {split_manifest['strategy']}, split_seed={split_seed}, "
        f"val_groups={split_manifest['val_groups']}"
    )
    if split_manifest["group_overlap"]:
        print(
            "  WARNING: train/val share track groups: "
            f"{split_manifest['group_overlap']}"
        )
    print(f"  model_seed: {model_seed}")
    print(
        f"  corpus: {data_summary['num_frames']} frames, "
        f"{data_summary['num_cones']} cones, sha256={split_manifest['dataset_sha256']}"
    )
    print(
        f"  train: {len(train_samples)} frames / "
        f"{sum(sample.points.shape[0] for sample in train_samples)} cones"
    )
    print(
        f"  val:   {len(val_samples)} frames / "
        f"{sum(sample.points.shape[0] for sample in val_samples)} cones"
    )

    architecture = model_cfg.get("architecture", "dgcnn")
    if architecture == "knn":
        metrics = run_knn_baseline(train_samples, val_samples, model_cfg)
        _print_metrics("k-NN validation", metrics)
        return
    if architecture == "lateral":
        metrics = run_lateral_baseline(
            val_samples, data_cfg.get("coordinate_frame", "")
        )
        _print_metrics("Lateral-sign validation", metrics)
        return

    model = build_model(model_cfg, data_cfg).to(device)
    print(
        f"  model: {architecture}, params={sum(p.numel() for p in model.parameters())}"
    )
    print(f"  device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(training_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    batch_size = int(training_cfg["batch_size"])
    normalize = data_cfg.get("normalize") in ("center", True)
    selection_metric = training_cfg.get("selection_metric", "macro_track_accuracy")

    best_value = float("-inf")
    best_epoch = -1
    best_state = None
    best_metrics = None
    metrics_path = save_dir / "metrics.jsonl"
    metrics_path.write_text("")

    for epoch in range(epochs):
        started = time.time()
        train_loss, train_accuracy = train_epoch(
            model,
            optimizer,
            train_samples,
            batch_size,
            augmentation_cfg,
            input_mode=data_cfg["input_mode"],
            normalize=normalize,
            device=device,
        )
        val_metrics = evaluate(
            model,
            val_samples,
            input_mode=data_cfg["input_mode"],
            normalize=normalize,
            device=device,
            num_classes=num_classes,
        )
        scheduler.step()

        if selection_metric not in val_metrics:
            raise KeyError(f"Unknown training.selection_metric: {selection_metric}")
        selected_value = float(val_metrics[selection_metric])
        improved = selected_value > best_value
        if improved:
            best_value = selected_value
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
            best_metrics = copy.deepcopy(val_metrics)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.time() - started,
            "is_best": improved,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"Epoch {epoch:03d}/{epochs} loss={train_loss:.4f} "
            f"train={train_accuracy:.4f} val={val_metrics['accuracy']:.4f} "
            f"balanced={val_metrics['balanced_accuracy']:.4f} "
            f"macro_track={val_metrics['macro_track_accuracy']:.4f} "
            f"worst={val_metrics['worst_track_accuracy']:.4f} "
            f"time={record['elapsed_seconds']:.1f}s{' *' if improved else ''}"
        )

    last_state = _state_dict_cpu(model)
    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")

    checkpoint_config = copy.deepcopy(config)
    checkpoint_config["data"]["data_dir"] = str(data_dir)
    checkpoint_config.setdefault("save", {})["save_dir"] = str(save_dir)
    common = {
        "checkpoint_version": 2,
        "config": checkpoint_config,
        "provenance": runtime_provenance(),
        "split_manifest": split_manifest,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_acc": best_metrics["accuracy"],
    }
    checkpoint_name = save_cfg.get("checkpoint_name", f"{architecture}.pth")
    best_path = save_dir / checkpoint_name
    suffix = best_path.suffix or ".pth"
    last_path = best_path.with_name(f"{best_path.stem}.last{suffix}")
    torch.save(
        {**common, "model_state": best_state, "checkpoint_role": "best"}, best_path
    )
    torch.save(
        {**common, "model_state": last_state, "checkpoint_role": "last"}, last_path
    )

    print(f"Best epoch: {best_epoch}, {selection_metric}={best_value:.4f}")
    _print_metrics("Best validation", best_metrics)
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved last checkpoint: {last_path}")
    print(f"Saved split manifest: {save_dir / 'split_manifest.json'}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
