"""Exercise the YAML workflow with a tiny, temporary three-family corpus."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sidenet-workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "corpus"
        self.data.mkdir()
        tracks = []
        for source, family in (("real", "TrainMap"), ("synthetic", "ValMap"), ("real", "TestMap")):
            for seed in (42, 43):
                name = f"{family}_s{seed}"
                tracks.append({"name": name})
                track = self.data / name
                track.mkdir()
                (track / "cloud_0.txt").write_text(
                    "1 2 0 0.2 0.2 0.3 0 1 Cone_Left\n"
                    "2 2 0 0.2 0.2 0.3 0 1 Cone_Left\n"
                    "1 -2 0 0.2 0.2 0.3 0 1 Cone_Right\n"
                    "2 -2 0 0.2 0.2 0.3 0 1 Cone_Right\n"
                )
                self.write_yaml(track / "metadata.yaml", {
                    "schema_version": 1, "track_name": name,
                    "coordinate_frame": "ego", "side_semantics": "track_global",
                    "frames": [{"frame_id": 0, "file": "cloud_0.txt"}],
                    "source": {"type": source, "track": family, "seed": seed},
                    "config": {"seed": seed},
                })
        self.write_yaml(self.data / "dataset_manifest.yaml", {
            "schema_version": 1, "coordinate_frame": "ego",
            "side_semantics": "track_global", "tracks": tracks,
        })
        self.prepared = self.root / "train_data.yaml"
        self.manifest = self.root / "three_way.json"
        self.split_config = self.root / "split.yaml"
        self.policy = {
            "data_dir": str(self.data),
            "output": {"data_config": str(self.prepared), "manifest": str(self.manifest)},
            "expected_seeds": [42, 43], "split_seed": 7,
            "real": {"train": ["TrainMap"], "validation": [], "test": ["TestMap"]},
            "synthetic": {"train": [], "validation": ["ValMap"], "test": []},
        }
        self.write_yaml(self.split_config, self.policy)
        self.training_config = self.root / "train.yaml"
        self.recipe = yaml.safe_load((PROJECT / "configs/train.yaml").read_text())
        self.recipe["data_config"] = str(self.prepared)
        self.recipe["training"].update(epochs=1, batch_size=2, device="cpu", lr=0.002)
        self.recipe["model"].update(dgcnn_k=2, dgcnn_dims=[8, 8])
        self.recipe["save"]["save_dir"] = str(self.root / "run")
        self.write_yaml(self.training_config, self.recipe)

    @staticmethod
    def write_yaml(path, value):
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    def prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            workflow.run_split(self.split_config)

    def command(self, script, config, *args):
        return subprocess.run(
            ["bash", str(PROJECT / script), "--config", str(config), *args],
            cwd=self.root, env=dict(os.environ, PYTHON_BIN=sys.executable, PYTHONDONTWRITEBYTECODE="1"),
            capture_output=True, text=True, timeout=60,
        )

    def test_split_writes_only_data_settings_and_keeps_families_together(self):
        original = self.training_config.read_bytes()
        self.prepare()
        self.prepare()  # Regenerating the split must preserve training edits.
        self.assertEqual(self.training_config.read_bytes(), original)
        prepared = yaml.safe_load(self.prepared.read_text())
        self.assertEqual(set(prepared), {"data", "split", "split_manifest"})
        self.assertEqual(prepared["split"]["seed"], 7)
        self.assertEqual(prepared["data"]["include_dirs"], ["TrainMap_s42", "TrainMap_s43", "ValMap_s42", "ValMap_s43"])
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(manifest["group_overlap"], [])
        self.assertEqual(manifest["partitions"]["test"]["tracks"], ["TestMap_s42", "TestMap_s43"])

    def test_shell_scripts_work_from_other_directory_and_preview_is_read_only(self):
        result = self.command("split_data.sh", self.split_config, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.prepared.exists())
        result = self.command("split_data.sh", self.split_config)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.command("start_training.sh", self.training_config, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lr: 0.002", result.stdout)
        self.assertFalse((self.root / "run").exists())

    def test_split_cannot_overwrite_handwritten_training_config(self):
        original = self.training_config.read_bytes()
        for output in ("data_config", "manifest"):
            with self.subTest(output=output):
                policy = yaml.safe_load(self.split_config.read_text())
                policy["output"][output] = str(self.training_config)
                path = self.root / "bad_split.yaml"
                self.write_yaml(path, policy)
                with self.assertRaises(ValueError):
                    workflow.run_split(path)
                self.assertEqual(self.training_config.read_bytes(), original)

    def test_legacy_prepare_still_outputs_full_training_config(self):
        from prepare_mixed_split import prepare
        legacy = self.root / "legacy.yaml"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(self.data, self.split_config, legacy, self.manifest)
        config = yaml.safe_load(legacy.read_text())
        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertNotIn("TestMap_s42", config["data"]["include_dirs"])

    def test_environment_is_set_before_training_process_and_settings_are_archived(self):
        self.prepare()
        with patch("workflow.subprocess.run") as launch, contextlib.redirect_stdout(io.StringIO()):
            workflow.run_training(self.training_config)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(launch.call_args.kwargs["env"]["OMP_NUM_THREADS"], "2")
        run = self.root / "run"
        self.assertEqual((run / "run_settings.yaml").read_bytes(), self.training_config.read_bytes())
        self.assertEqual((run / "dataset_split.json").read_bytes(), self.manifest.read_bytes())
        resolved = yaml.safe_load((run / "resolved_config.yaml").read_text())
        self.assertEqual(resolved["training"]["lr"], 0.002)
        self.assertNotIn("TestMap_s42", resolved["data"]["include_dirs"])

    def test_existing_training_results_are_not_overwritten(self):
        self.prepare()
        run = self.root / "run"
        run.mkdir()
        checkpoint = run / "dgcnn.pth"
        checkpoint.write_bytes(b"preserve previous model")
        with self.assertRaises(FileExistsError):
            workflow.run_training(self.training_config)
        self.assertEqual(checkpoint.read_bytes(), b"preserve previous model")

    def test_training_cannot_override_partitions_or_use_tampered_prepared_config(self):
        self.prepare()
        self.recipe["data"]["include_dirs"] = ["TestMap_s42"]
        self.write_yaml(self.training_config, self.recipe)
        with self.assertRaisesRegex(ValueError, "may only change"):
            workflow.resolve_training(self.training_config)
        del self.recipe["data"]["include_dirs"]
        self.write_yaml(self.training_config, self.recipe)
        prepared = yaml.safe_load(self.prepared.read_text())
        prepared["data"]["include_dirs"].append("TestMap_s42")
        self.write_yaml(self.prepared, prepared)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            workflow.resolve_training(self.training_config)

    def test_changed_frames_fail_before_training(self):
        self.prepare()
        frame = self.data / "TrainMap_s42/cloud_0.txt"
        frame.write_text(frame.read_text().replace("1 2 0", "9 2 0"))
        result = self.command("start_training.sh", self.training_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Dataset changed since preparation", result.stderr)
        self.assertFalse((self.root / "run/metrics.jsonl").exists())

    def test_tiny_cpu_training_produces_checkpoints_and_excludes_test_frames(self):
        self.prepare()
        result = self.command("start_training.sh", self.training_config)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = self.root / "run"
        self.assertTrue((run / "dgcnn.pth").is_file())
        self.assertTrue((run / "dgcnn.last.pth").is_file())
        actual = json.loads((run / "split_manifest.json").read_text())
        self.assertFalse(any("TestMap" in name for name in actual["train_frames"] + actual["val_frames"]))
        self.assertEqual(len((run / "metrics.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
