"""Data contracts, loading, augmentation, and grouped splitting for SideNet."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import yaml


INPUT_FEATURES = {
    "xy": (0, 1),
    "xyz": (0, 1, 2),
    "xyzs": (0, 1, 2, 3),
}
LABEL_TO_ID = {"Left": 0, "Right": 1, "Unknown": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass
class FrameSample:
    points: torch.Tensor
    labels: Optional[torch.Tensor]
    track: str
    frame_id: int
    path: Path
    coordinate_frame: str
    side_semantics: str

    @property
    def sample_id(self) -> str:
        return f"{self.track}/{self.path.name}"


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch from one recorded value."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def select_input_features(points: torch.Tensor, input_mode: str) -> torch.Tensor:
    """Select the exact channels promised by ``input_mode``."""
    if input_mode not in INPUT_FEATURES:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return points[..., list(INPUT_FEATURES[input_mode])]


def normalize_frame(points: torch.Tensor) -> torch.Tensor:
    """Center x/y at the observed-set centroid (legacy ablation only)."""
    points = points.clone()
    if points.shape[0] > 0:
        points[:, :2] -= points[:, :2].mean(dim=0, keepdim=True)
    return points


def augment_frame(
    points: torch.Tensor,
    labels: torch.Tensor,
    *,
    rot_range: float = 0.2618,
    scale_range: Sequence[float] = (0.97, 1.03),
    noise_std: float = 0.02,
    flip_prob: float = 0.5,
    transform_origin: str = "ego",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply proper SE(2) jitter and optional reflection with label permutation.

    ``transform_origin='ego'`` is the correct default for ego/LiDAR-frame input.
    ``'centroid'`` is retained only for reproducing the legacy world-frame run.
    """
    if transform_origin not in {"ego", "centroid"}:
        raise ValueError("transform_origin must be 'ego' or 'centroid'")
    points = points.clone()
    labels = labels.clone()
    if points.shape[0] == 0:
        return points, labels

    origin = (
        points[:, :2].mean(dim=0)
        if transform_origin == "centroid"
        else points.new_zeros(2)
    )
    xy = points[:, :2] - origin

    angle = (torch.rand((), device=points.device).item() * 2.0 - 1.0) * rot_range
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rotation = points.new_tensor([[cos_a, -sin_a], [sin_a, cos_a]])
    xy = xy @ rotation.T

    low, high = float(scale_range[0]), float(scale_range[1])
    scale = torch.empty((), device=points.device).uniform_(low, high)
    points[:, :2] = xy * scale + origin
    if noise_std > 0:
        points[:, :2] += torch.randn_like(points[:, :2]) * noise_std

    if torch.rand((), device=points.device).item() < flip_prob:
        points[:, 1] = -points[:, 1]
        left = labels == LABEL_TO_ID["Left"]
        right = labels == LABEL_TO_ID["Right"]
        labels[left] = LABEL_TO_ID["Right"]
        labels[right] = LABEL_TO_ID["Left"]

    return points, labels


def _frame_id(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Frame filename has no numeric suffix: {path}")
    return int(match.group(1))


def _sorted_cloud_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("cloud_*.txt"), key=_frame_id)


def _read_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _manifest_track_names(manifest: dict) -> list[str]:
    tracks = manifest.get("tracks", [])
    names = []
    for item in tracks:
        names.append(str(item["name"] if isinstance(item, dict) else item))
    return names


def _resolve_track_dirs(
    data_dir: Path,
    include_dirs: Optional[Sequence[str]],
    exclude_dirs: Optional[Sequence[str]],
    allow_legacy_without_metadata: bool,
) -> tuple[list[Path], dict]:
    root_manifest_path = data_dir / "dataset_manifest.yaml"
    root_manifest = (
        _read_yaml(root_manifest_path) if root_manifest_path.exists() else {}
    )

    if (data_dir / "cloud_0.txt").exists():
        return [data_dir], root_manifest

    if include_dirs:
        names = [str(name) for name in include_dirs]
    elif root_manifest:
        names = _manifest_track_names(root_manifest)
    elif allow_legacy_without_metadata:
        names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    else:
        raise FileNotFoundError(
            f"{root_manifest_path} is missing. Regenerate data with bitfsd-generator "
            "or explicitly enable data.allow_legacy_without_metadata."
        )

    excluded = tuple(exclude_dirs or ())
    directories = []
    for name in names:
        if any(pattern in name for pattern in excluded):
            continue
        path = data_dir / name
        if not path.is_dir():
            raise FileNotFoundError(
                f"Dataset manifest/include_dirs references missing track: {path}"
            )
        directories.append(path)
    return directories, root_manifest


def parse_cone_line(line: str, require_labels: bool = True):
    """Parse legacy or v1 SideNet text rows.

    Canonical v1 rows are ``x y z dx dy dz heading score class_name``. For
    detector-only inference, rows without a side label are retained.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if len(parts) < 3:
        raise ValueError(f"Expected at least x y z, got: {line!r}")

    x, y, z = map(float, parts[:3])
    tail = parts[-1]
    label = None
    for name, label_id in LABEL_TO_ID.items():
        if name in tail:
            label = label_id
            break
    if require_labels and label is None:
        return None

    score = 1.0
    if len(parts) >= 9:
        try:
            score = float(parts[-2])
        except ValueError:
            pass
    elif len(parts) == 4:
        try:
            score = float(parts[3])
        except ValueError:
            pass
    return [x, y, z, score], label


def load_frames(
    data_dir: str | Path,
    *,
    include_dirs: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    expected_coordinate_frame: Optional[str] = None,
    expected_side_semantics: Optional[str] = None,
    allow_legacy_without_metadata: bool = False,
    assumed_coordinate_frame: Optional[str] = None,
    require_labels: bool = True,
) -> tuple[list[FrameSample], dict]:
    """Load a manifest-backed dataset without silently ingesting stale folders."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    track_dirs, root_manifest = _resolve_track_dirs(
        data_dir, include_dirs, exclude_dirs, allow_legacy_without_metadata
    )
    if root_manifest:
        if root_manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported dataset manifest schema: "
                f"{root_manifest.get('schema_version')!r}"
            )
        root_frame = root_manifest.get("coordinate_frame")
        root_side = root_manifest.get("side_semantics")
        if expected_coordinate_frame and root_frame != expected_coordinate_frame:
            raise ValueError(
                f"Root manifest coordinate_frame={root_frame!r}, "
                f"expected={expected_coordinate_frame!r}"
            )
        if expected_side_semantics and root_side != expected_side_semantics:
            raise ValueError(
                f"Root manifest side_semantics={root_side!r}, "
                f"expected={expected_side_semantics!r}"
            )
    samples: list[FrameSample] = []
    n_files = 0
    skipped_unlabeled = 0

    for track_dir in track_dirs:
        metadata_path = track_dir / "metadata.yaml"
        metadata = {}
        if metadata_path.exists():
            metadata = _read_yaml(metadata_path)
            if metadata.get("schema_version") != 1:
                raise ValueError(
                    f"Unsupported track metadata schema in {metadata_path}: "
                    f"{metadata.get('schema_version')!r}"
                )
            if metadata.get("track_name") not in {None, track_dir.name}:
                raise ValueError(
                    f"Track metadata name mismatch: directory={track_dir.name!r}, "
                    f"metadata={metadata.get('track_name')!r}"
                )
            coordinate_frame = metadata.get("coordinate_frame")
            side_semantics = metadata.get("side_semantics")
        elif allow_legacy_without_metadata:
            coordinate_frame = assumed_coordinate_frame or "world"
            side_semantics = expected_side_semantics or "track_global"
        else:
            raise FileNotFoundError(
                f"Missing coordinate/label contract: {metadata_path}. "
                "Regenerate the dataset or explicitly enable legacy mode."
            )

        if expected_coordinate_frame and coordinate_frame != expected_coordinate_frame:
            raise ValueError(
                f"Coordinate mismatch for {track_dir.name}: dataset={coordinate_frame!r}, "
                f"expected={expected_coordinate_frame!r}"
            )
        if expected_side_semantics and side_semantics != expected_side_semantics:
            raise ValueError(
                f"Side-semantics mismatch for {track_dir.name}: dataset={side_semantics!r}, "
                f"expected={expected_side_semantics!r}"
            )

        if metadata.get("frames"):
            cloud_files = [track_dir / frame["file"] for frame in metadata["frames"]]
            missing_files = [path for path in cloud_files if not path.is_file()]
            if missing_files:
                raise FileNotFoundError(
                    f"Track metadata references missing frame files: {missing_files[:3]}"
                )
        else:
            cloud_files = _sorted_cloud_files(track_dir)
        n_files += len(cloud_files)
        for cloud_file in cloud_files:
            points = []
            labels = []
            for line_number, line in enumerate(
                cloud_file.read_text().splitlines(), start=1
            ):
                try:
                    parsed = parse_cone_line(line, require_labels=require_labels)
                except ValueError as exc:
                    raise ValueError(f"{cloud_file}:{line_number}: {exc}") from exc
                if parsed is None:
                    skipped_unlabeled += 1
                    continue
                point, label = parsed
                points.append(point)
                if label is not None:
                    labels.append(label)

            if not points:
                continue
            label_tensor = None
            if labels:
                if len(labels) != len(points):
                    raise ValueError(f"Mixed labeled/unlabeled rows in {cloud_file}")
                label_tensor = torch.tensor(labels, dtype=torch.long)
            samples.append(
                FrameSample(
                    points=torch.tensor(points, dtype=torch.float32),
                    labels=label_tensor,
                    track=track_dir.name,
                    frame_id=_frame_id(cloud_file),
                    path=cloud_file,
                    coordinate_frame=str(coordinate_frame),
                    side_semantics=str(side_semantics),
                )
            )

    summary = {
        "data_dir": str(data_dir),
        "root_manifest": root_manifest,
        "num_track_dirs": len(track_dirs),
        "tracks": [path.name for path in track_dirs],
        "num_files": n_files,
        "num_frames": len(samples),
        "num_cones": sum(sample.points.shape[0] for sample in samples),
        "skipped_unlabeled_rows": skipped_unlabeled,
    }
    return samples, summary


def canonical_track_group(track_name: str) -> str:
    """Keep derived variants of one physical track in the same split."""
    name = track_name
    changed = True
    while changed:
        previous = name
        name = re.sub(r"_(?:flip|test)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"_V\d+$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"_s\d+$", "", name, flags=re.IGNORECASE)
        changed = name != previous
    return name


def dataset_fingerprint(samples: Iterable[FrameSample]) -> str:
    """Hash sample identities and bytes so a result names the exact corpus."""
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item.sample_id):
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.coordinate_frame.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.side_semantics.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def split_frames(samples: Sequence[FrameSample], split_cfg: dict):
    """Create a deterministic random-frame or track-family holdout split."""
    if len(samples) < 2:
        raise ValueError("At least two non-empty frames are required")
    strategy = split_cfg.get("strategy", "track_holdout")
    seed = int(split_cfg.get("seed", 42))
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    if strategy == "random_frame":
        order = torch.randperm(len(samples), generator=generator).tolist()
        n_val = max(1, int(len(samples) * val_ratio))
        val_indices, train_indices = order[:n_val], order[n_val:]
        val_groups = sorted(
            {canonical_track_group(samples[i].track) for i in val_indices}
        )
    elif strategy == "track_holdout":
        groups = sorted({canonical_track_group(sample.track) for sample in samples})
        configured = split_cfg.get("val_groups") or []
        if configured:
            val_groups = [str(group) for group in configured]
            missing = sorted(set(val_groups) - set(groups))
            if missing:
                raise ValueError(
                    f"Unknown validation track groups: {missing}; available={groups}"
                )
        else:
            order = torch.randperm(len(groups), generator=generator).tolist()
            n_val_groups = max(1, int(round(len(groups) * val_ratio)))
            val_groups = [groups[i] for i in order[:n_val_groups]]
        val_group_set = set(val_groups)
        val_indices = [
            i
            for i, sample in enumerate(samples)
            if canonical_track_group(sample.track) in val_group_set
        ]
        val_index_set = set(val_indices)
        train_indices = [i for i in range(len(samples)) if i not in val_index_set]
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if not train_indices or not val_indices:
        raise ValueError(
            f"Split produced an empty partition: train={len(train_indices)}, val={len(val_indices)}"
        )
    train = [samples[i] for i in train_indices]
    val = [samples[i] for i in val_indices]
    train_groups = sorted({canonical_track_group(sample.track) for sample in train})
    if strategy == "track_holdout" and set(train_groups) & set(val_groups):
        raise AssertionError("Track-family leakage detected in grouped split")

    manifest = {
        "schema_version": 1,
        "strategy": strategy,
        "seed": seed,
        "val_ratio": val_ratio,
        "dataset_sha256": dataset_fingerprint(samples),
        "train_groups": train_groups,
        "val_groups": sorted(val_groups),
        "group_overlap": sorted(set(train_groups) & set(val_groups)),
        "train_frames": [sample.sample_id for sample in train],
        "val_frames": [sample.sample_id for sample in val],
    }
    return train, val, manifest


def write_split_manifest(manifest: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
