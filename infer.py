#!/usr/bin/env python3
"""Evaluate a SideNet checkpoint or classify unlabeled detector outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sidenet import build_model
from sidenet_data import (
    FrameSample,
    ID_TO_LABEL,
    dataset_fingerprint,
    load_frames,
    normalize_frame,
    select_input_features,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _legacy_runtime_config(checkpoint_config: dict) -> dict:
    """Lift v1 checkpoint metadata into the v2 nested config contract."""
    architecture = checkpoint_config.get("architecture", "transformer")
    return {
        "data": {
            "input_mode": "xyzs",
            "num_classes": 2,
            "coordinate_frame": "world",
            "side_semantics": "track_global",
            "normalize": None,
        },
        "model": {**checkpoint_config, "architecture": architecture},
        "legacy_checkpoint": True,
    }


def load_checkpoint(path: str | Path, device: str | torch.device = "auto"):
    device = _resolve_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_config = checkpoint.get("config", {})
    if "data" in raw_config and "model" in raw_config:
        runtime_config = raw_config
        runtime_config["legacy_checkpoint"] = False
    else:
        runtime_config = _legacy_runtime_config(raw_config)

    model = build_model(runtime_config["model"], runtime_config["data"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    architecture = runtime_config["model"].get("architecture", "unknown")
    role = checkpoint.get("checkpoint_role", "legacy/final")
    print(
        f"Loaded {architecture} ({role}, best val micro: "
        f"{checkpoint.get('best_acc', '?')}) on {device}"
    )
    return model, runtime_config, checkpoint, device


class SideNetPredictor:
    """Reusable adapter for OpenPCDet-style ``xyz + pred_scores`` arrays."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.model, self.config, self.checkpoint, self.device = load_checkpoint(
            checkpoint, device
        )
        self.data_config = self.config["data"]

    @torch.no_grad()
    def predict_tensor(self, points: torch.Tensor):
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError("points must have shape (N, 3) or (N, 4)")
        if points.shape[0] == 0:
            num_classes = int(self.data_config.get("num_classes", 2))
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0),
                torch.empty(0, num_classes),
            )
        if points.shape[1] == 3:
            points = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points = points.to(self.device, dtype=torch.float32)
        if self.data_config.get("normalize") in ("center", True):
            points = normalize_frame(points)
        model_input = select_input_features(points, self.data_config["input_mode"])
        probabilities = self.model(model_input, return_logits=False)
        confidence, predictions = probabilities.max(dim=-1)
        return predictions.cpu(), confidence.cpu(), probabilities.cpu()

    def predict_detected_cones(self, xyz, scores=None, coordinate_frame="ego"):
        """Predict detector boxes after an explicit coordinate-frame assertion."""
        expected_frame = self.data_config.get("coordinate_frame")
        if expected_frame and coordinate_frame != expected_frame:
            raise ValueError(
                f"Checkpoint expects {expected_frame!r} coordinates, got "
                f"{coordinate_frame!r}"
            )
        xyz_tensor = torch.as_tensor(xyz, dtype=torch.float32)
        if xyz_tensor.numel() == 0:
            xyz_tensor = xyz_tensor.reshape(0, 3)
        if xyz_tensor.ndim != 2 or xyz_tensor.shape[1] < 3:
            raise ValueError("xyz must have shape (N, 3+) with x/y/z first")
        if scores is None:
            score_tensor = torch.ones(xyz_tensor.shape[0], 1)
        else:
            score_tensor = torch.as_tensor(scores, dtype=torch.float32).reshape(-1, 1)
        if score_tensor.shape[0] != xyz_tensor.shape[0]:
            raise ValueError("xyz and scores must have the same number of detections")
        return self.predict_tensor(torch.cat([xyz_tensor[:, :3], score_tensor], dim=-1))


def _predict_sample(predictor: SideNetPredictor, sample: FrameSample, threshold: float):
    predictions, confidence, probabilities = predictor.predict_tensor(sample.points)
    accepted = confidence >= threshold
    return predictions, confidence, probabilities, accepted


def evaluate_labeled(
    predictor: SideNetPredictor, samples: list[FrameSample], threshold: float
) -> dict:
    num_classes = int(predictor.data_config.get("num_classes", 2))
    correct = total = 0
    abstained = 0
    frame_results = []
    track_counts = defaultdict(lambda: [0, 0])
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    true_support = torch.zeros(num_classes, dtype=torch.long)
    predicted_support = torch.zeros(num_classes, dtype=torch.long)
    true_positive = torch.zeros(num_classes, dtype=torch.long)
    nll_sum = 0.0
    calibration_count = torch.zeros(15, dtype=torch.long)
    calibration_confidence = torch.zeros(15, dtype=torch.float64)
    calibration_correct = torch.zeros(15, dtype=torch.float64)

    for sample in samples:
        predictions, confidence, probabilities, accepted = _predict_sample(
            predictor, sample, threshold
        )
        labels = sample.labels
        is_correct = (predictions == labels) & accepted
        frame_correct = is_correct.sum().item()
        frame_total = labels.numel()
        correct += frame_correct
        total += frame_total
        abstained += (~accepted).sum().item()
        frame_results.append((sample.sample_id, frame_correct / frame_total))
        track_counts[sample.track][0] += frame_correct
        track_counts[sample.track][1] += frame_total
        true_support += torch.bincount(labels, minlength=num_classes)
        if accepted.any():
            predicted_support += torch.bincount(
                predictions[accepted], minlength=num_classes
            )
            for class_id in range(num_classes):
                true_positive[class_id] += (
                    (labels == class_id) & (predictions == class_id) & accepted
                ).sum()
            flat = labels[accepted] * num_classes + predictions[accepted]
            confusion += torch.bincount(flat, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )
        nll_sum += (
            -probabilities[torch.arange(labels.numel()), labels]
            .clamp(min=1e-12)
            .log()
            .sum()
            .item()
        )
        bins = (confidence * 15).long().clamp(max=14)
        calibration_count += torch.bincount(bins, minlength=15)
        calibration_confidence += torch.bincount(
            bins, weights=confidence.double(), minlength=15
        )
        calibration_correct += torch.bincount(
            bins,
            weights=(predictions == labels).double(),
            minlength=15,
        )

    per_track = {
        track: values[0] / values[1] for track, values in sorted(track_counts.items())
    }
    recall = true_positive.float() / true_support.clamp(min=1).float()
    precision = true_positive.float() / predicted_support.clamp(min=1).float()
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    present = true_support > 0
    populated = calibration_count > 0
    bin_accuracy = calibration_correct[populated] / calibration_count[populated]
    bin_confidence = calibration_confidence[populated] / calibration_count[populated]
    ece = (
        ((bin_accuracy - bin_confidence).abs() * calibration_count[populated]).sum()
        / total
    ).item()
    return {
        "accuracy_with_abstentions_as_errors": correct / total,
        "balanced_accuracy": recall[present].mean().item(),
        "macro_f1": f1[present].mean().item(),
        "nll": nll_sum / total,
        "ece_15_bin": ece,
        "coverage": 1.0 - abstained / total,
        "macro_frame_accuracy": float(np.mean([value for _, value in frame_results])),
        "macro_track_accuracy": float(np.mean(list(per_track.values()))),
        "worst_track_accuracy": min(per_track.values()),
        "per_track_accuracy": per_track,
        "confusion_on_accepted": confusion.tolist(),
        "worst_frames": sorted(frame_results, key=lambda item: item[1])[:5],
        "num_frames": len(samples),
        "num_cones": total,
    }


def write_unlabeled_predictions(
    predictor: SideNetPredictor,
    samples: list[FrameSample],
    threshold: float,
    output_dir: Path | None,
) -> None:
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        predictions, confidence, probabilities, accepted = _predict_sample(
            predictor, sample, threshold
        )
        names = [
            ID_TO_LABEL.get(int(prediction), f"Class_{int(prediction)}")
            if bool(is_accepted)
            else "Unknown"
            for prediction, is_accepted in zip(predictions, accepted)
        ]
        counts = {name: names.count(name) for name in ("Left", "Right", "Unknown")}
        print(
            f"{sample.sample_id}: {len(names)} cones -> "
            f"{counts['Left']} Left, {counts['Right']} Right, "
            f"{counts['Unknown']} abstained"
        )
        if output_dir:
            probability_names = [
                f"p_{ID_TO_LABEL.get(index, f'class_{index}').lower()}"
                for index in range(probabilities.shape[1])
            ]
            lines = [
                "# x y z detector_score predicted_side confidence "
                + " ".join(probability_names)
            ]
            for point, name, conf, probs in zip(
                sample.points, names, confidence, probabilities
            ):
                lines.append(
                    f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                    f"{point[3]:.4f} {name} {conf:.6f} "
                    + " ".join(f"{probability:.6f}" for probability in probs)
                )
            target = output_dir / sample.track / sample.path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-gt", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--coordinate-frame",
        choices=["ego", "world"],
        help="Required assumption for detector text without metadata.yaml",
    )
    parser.add_argument(
        "--allow-legacy-data",
        action="store_true",
        help="Allow input without dataset/track metadata",
    )
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON")
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Evaluate every labeled frame instead of the checkpoint validation split",
    )
    parser.add_argument(
        "--allow-dataset-mismatch",
        action="store_true",
        help="Do not fail when the current corpus hash differs from the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1]")
    predictor = SideNetPredictor(args.ckpt, args.device)
    expected_frame = predictor.data_config.get("coordinate_frame")
    assumed_frame = args.coordinate_frame or expected_frame
    if (
        args.coordinate_frame
        and expected_frame
        and args.coordinate_frame != expected_frame
    ):
        raise ValueError(
            f"Checkpoint expects {expected_frame!r} coordinates, but CLI declares "
            f"{args.coordinate_frame!r}. Convert coordinates instead of bypassing the contract."
        )

    samples, summary = load_frames(
        args.data_dir,
        include_dirs=predictor.data_config.get("include_dirs"),
        exclude_dirs=predictor.data_config.get("exclude_dirs"),
        expected_coordinate_frame=expected_frame,
        expected_side_semantics=predictor.data_config.get("side_semantics"),
        allow_legacy_without_metadata=(
            args.allow_legacy_data
            or predictor.config.get("legacy_checkpoint", False)
            or predictor.data_config.get("allow_legacy_without_metadata", False)
        ),
        assumed_coordinate_frame=assumed_frame,
        require_labels=not args.no_gt,
    )
    if not args.no_gt:
        split_manifest = predictor.checkpoint.get("split_manifest")
        if split_manifest:
            current_hash = dataset_fingerprint(samples)
            expected_hash = split_manifest.get("dataset_sha256")
            if (
                expected_hash
                and current_hash != expected_hash
                and not args.allow_dataset_mismatch
            ):
                raise ValueError(
                    "Dataset SHA-256 does not match the checkpoint split manifest. "
                    "Use the exact corpus, or pass --allow-dataset-mismatch only for diagnostics."
                )
            if not args.all_frames:
                wanted = set(split_manifest.get("val_frames", []))
                samples = [sample for sample in samples if sample.sample_id in wanted]
                found = {sample.sample_id for sample in samples}
                missing = sorted(wanted - found)
                if missing:
                    raise ValueError(
                        f"Checkpoint validation split references {len(missing)} missing frames"
                    )
                print(
                    f"Evaluation scope: checkpoint validation split "
                    f"({len(samples)} frames)"
                )
    print(
        f"Loaded corpus: {summary['num_frames']} frames / {summary['num_cones']} cones; "
        f"evaluating {len(samples)} frames / "
        f"{sum(sample.points.shape[0] for sample in samples)} cones; "
        f"coordinate_frame={expected_frame}"
    )

    if args.no_gt:
        write_unlabeled_predictions(
            predictor, samples, args.min_confidence, args.output_dir
        )
        return

    metrics = evaluate_labeled(predictor, samples, args.min_confidence)
    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    print(
        f"Overall: {metrics['accuracy_with_abstentions_as_errors']:.4f}, "
        f"balanced={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, "
        f"coverage={metrics['coverage']:.4f}, "
        f"macro_frame={metrics['macro_frame_accuracy']:.4f}, "
        f"macro_track={metrics['macro_track_accuracy']:.4f}, "
        f"worst_track={metrics['worst_track_accuracy']:.4f}"
    )
    print(f"Calibration: NLL={metrics['nll']:.4f}, ECE15={metrics['ece_15_bin']:.4f}")
    print(f"Confusion on accepted: {metrics['confusion_on_accepted']}")
    for track, accuracy in metrics["per_track_accuracy"].items():
        print(f"  {track}: {accuracy:.4f}")
    print("Worst 5 frames:")
    for sample_id, accuracy in metrics["worst_frames"]:
        print(f"  {accuracy:.3f}  {sample_id}")


if __name__ == "__main__":
    main()
