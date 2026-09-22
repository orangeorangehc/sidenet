import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from infer import SideNetPredictor
from sidenet_data import (
    FrameSample,
    augment_frame,
    canonical_track_group,
    load_frames,
    parse_cone_line,
    select_input_features,
    split_frames,
)
from train_sidenet import load_config
from train_sidenet import evaluate_baseline


def _write_track(root: Path, name: str, frame_ids=(0, 1), coordinate_frame="ego"):
    track = root / name
    track.mkdir(parents=True)
    for frame_id in frame_ids:
        (track / f"cloud_{frame_id}.txt").write_text(
            "1.0 2.0 0.0 0.2 0.2 0.3 0.0 0.9 Cone_Left\n"
            "1.0 -2.0 0.0 0.2 0.2 0.3 0.0 0.8 Cone_Right\n"
        )
    (track / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "track_name": name,
                "coordinate_frame": coordinate_frame,
                "side_semantics": "track_global",
                "frames": [
                    {"frame_id": frame_id, "file": f"cloud_{frame_id}.txt"}
                    for frame_id in frame_ids
                ],
            }
        )
    )


def _write_manifest(root: Path, names):
    (root / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "coordinate_frame": "ego",
                "side_semantics": "track_global",
                "tracks": [{"name": name} for name in names],
            }
        )
    )


def test_manifest_is_source_of_truth_and_stale_folders_are_ignored(tmp_path):
    _write_track(tmp_path, "TrackA")
    _write_track(tmp_path, "StaleTrack")
    (tmp_path / "TrackA" / "cloud_99.txt").write_text(
        "9 9 0 0.2 0.2 0.3 0 1 Cone_Left\n"
    )
    _write_manifest(tmp_path, ["TrackA"])
    samples, summary = load_frames(
        tmp_path,
        expected_coordinate_frame="ego",
        expected_side_semantics="track_global",
    )
    assert summary["tracks"] == ["TrackA"]
    assert len(samples) == 2
    assert all(sample.track == "TrackA" for sample in samples)


def test_coordinate_contract_fails_closed(tmp_path):
    _write_track(tmp_path, "TrackA", coordinate_frame="world")
    _write_manifest(tmp_path, ["TrackA"])
    with pytest.raises(ValueError, match="Coordinate mismatch"):
        load_frames(tmp_path, expected_coordinate_frame="ego")


def test_legacy_data_requires_explicit_opt_in(tmp_path):
    track = tmp_path / "TrackA"
    track.mkdir()
    (track / "cloud_0.txt").write_text("1 2 0 0.2 0.2 0.3 0 1.0 Cone_Left\n")
    with pytest.raises(FileNotFoundError, match="dataset_manifest"):
        load_frames(tmp_path)
    samples, _ = load_frames(
        tmp_path,
        include_dirs=["TrackA"],
        allow_legacy_without_metadata=True,
        assumed_coordinate_frame="world",
        expected_coordinate_frame="world",
    )
    assert len(samples) == 1


def test_track_family_holdout_keeps_variants_together(tmp_path):
    samples = []
    for track in ["FSE22", "FSE22_test", "FSS22_V1", "FSS22_V2", "FSG24"]:
        path = tmp_path / track / "cloud_0.txt"
        path.parent.mkdir()
        path.write_text("placeholder")
        samples.append(
            FrameSample(
                points=torch.randn(2, 4),
                labels=torch.tensor([0, 1]),
                track=track,
                frame_id=0,
                path=path,
                coordinate_frame="ego",
                side_semantics="track_global",
            )
        )
    train, val, manifest = split_frames(
        samples,
        {
            "strategy": "track_holdout",
            "seed": 42,
            "val_ratio": 0.2,
            "val_groups": ["FSE22", "FSS22"],
        },
    )
    assert {sample.track for sample in val} == {
        "FSE22",
        "FSE22_test",
        "FSS22_V1",
        "FSS22_V2",
    }
    assert {sample.track for sample in train} == {"FSG24"}
    assert set(manifest["train_groups"]).isdisjoint(manifest["val_groups"])
    assert manifest["group_overlap"] == []


@pytest.mark.parametrize(
    "name,expected",
    [
        ("FSE22_test_flip", "FSE22"),
        ("FSS22_V2", "FSS22"),
        ("hairpin_s42", "hairpin"),
        ("FSS22_V1_s42", "FSS22"),
        ("FSS22_V2_s44_flip", "FSS22"),
        ("FSE22_test_s43", "FSE22"),
        ("FSE22_s42_test_flip", "FSE22"),
    ],
)
def test_track_group_normalization(name, expected):
    assert canonical_track_group(name) == expected


def test_multiseed_holdout_keeps_all_versions_together(tmp_path):
    names = ["FSS22_V1_s42", "FSS22_V2_s43", "FSE22_s42", "FSE22_test_s44", "hairpin_s42"]
    for name in names:
        _write_track(tmp_path, name)
    _write_manifest(tmp_path, names)
    samples, _ = load_frames(tmp_path, expected_coordinate_frame="ego")
    train, val, manifest = split_frames(
        samples, {"strategy": "track_holdout", "val_groups": ["FSS22", "FSE22"]}
    )
    assert {s.track for s in train} == {"hairpin_s42"}
    assert {s.track for s in val} == set(names[:-1])
    assert manifest["group_overlap"] == []


def test_reflection_swaps_only_left_right_labels():
    points = torch.tensor([[1.0, 2.0, 0.0, 1.0], [1.0, -3.0, 0.0, 1.0]])
    labels = torch.tensor([0, 1])
    reflected, swapped = augment_frame(
        points,
        labels,
        rot_range=0.0,
        scale_range=[1.0, 1.0],
        noise_std=0.0,
        flip_prob=1.0,
        transform_origin="ego",
    )
    assert torch.equal(reflected[:, 1], torch.tensor([-2.0, 3.0]))
    assert torch.equal(swapped, torch.tensor([1, 0]))


def test_input_mode_selects_real_dimensions():
    points = torch.randn(4, 4)
    assert select_input_features(points, "xy").shape == (4, 2)
    assert select_input_features(points, "xyz").shape == (4, 3)
    assert select_input_features(points, "xyzs").shape == (4, 4)


def test_unlabeled_detector_row_is_retained_in_no_gt_mode():
    point, label = parse_cone_line(
        "1 2 3 0.2 0.2 0.3 0.0 0.73 Cone", require_labels=False
    )
    assert point == [1.0, 2.0, 3.0, 0.73]
    assert label is None


def test_detector_adapter_handles_empty_detection_frame():
    predictor = SideNetPredictor.__new__(SideNetPredictor)
    predictor.data_config = {"num_classes": 2, "coordinate_frame": "ego"}
    predictions, confidence, probabilities = predictor.predict_tensor(torch.empty(0, 3))
    assert predictions.shape == (0,)
    assert confidence.shape == (0,)
    assert probabilities.shape == (0, 2)
    with pytest.raises(ValueError, match="Checkpoint expects"):
        predictor.predict_detected_cones([], coordinate_frame="world")


def test_cli_overrides_update_nested_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "training": {"epochs": 10, "lr": 0.1, "seed": 1},
                "data": {"data_dir": "old"},
                "split": {"seed": 2},
            }
        )
    )
    config = load_config(
        path, {"epochs": 3, "data_dir": "new", "seed": 7, "split_seed": 11}
    )
    assert config["training"]["epochs"] == 3
    assert config["training"]["seed"] == 7
    assert config["split"]["seed"] == 11
    assert config["data"]["data_dir"] == "new"


def _load_generator_export_module():
    path = (
        Path(__file__).resolve().parents[2] / "bitfsd-generator" / "src" / "export.py"
    )
    spec = importlib.util.spec_from_file_location("bitfsd_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generator_perception_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "bitfsd-generator"
        / "src"
        / "perception.py"
    )
    spec = importlib.util.spec_from_file_location("bitfsd_perception", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_world_to_ego_canonicalization():
    export = _load_generator_export_module()
    x, y, z = export.world_to_ego_xyz(10.0, 2.0, 3.0, [10.0, 0.0, 1.0, 0.0])
    assert (x, y, z) == pytest.approx((0.0, 2.0, 2.0))


def test_baseline_metrics_include_balanced_accuracy_and_macro_f1(tmp_path):
    path = tmp_path / "TrackA" / "cloud_0.txt"
    path.parent.mkdir()
    path.write_text("placeholder")
    sample = FrameSample(
        points=torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        track="TrackA",
        frame_id=0,
        path=path,
        coordinate_frame="ego",
        side_semantics="track_global",
    )
    metrics = evaluate_baseline([sample], lambda frame: frame.labels.clone())
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_reverse_driving_side_semantics_are_explicit():
    perception = _load_generator_perception_module()

    class Track:
        name = "contract_test"
        left_xyz = np.array([[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
        right_xyz = np.array([[1.0, -2.0, 0.0]])
        centerline = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

    config = perception.PerceptionConfig(
        ego_spacing=100.0,
        ego_noise_x=0.0,
        ego_noise_y=0.0,
        ego_noise_yaw_deg=0.0,
        lidar_range_max=100.0,
        lidar_fov_deg=360.0,
        position_noise_xy=0.0,
        position_noise_z=0.0,
        drop_rate=0.0,
        fp_rate=0.0,
    )
    track_global = perception.PerceptionPipeline(config).run(
        Track(), flip=True, side_semantics="track_global"
    )
    ego_relative = perception.PerceptionPipeline(config).run(
        Track(), flip=True, side_semantics="ego_relative"
    )
    assert [cone.side for cone in track_global.frames[0].cones] == [
        "Left",
        "Left",
        "Right",
    ]
    assert [cone.side for cone in ego_relative.frames[0].cones] == [
        "Right",
        "Right",
        "Left",
    ]
