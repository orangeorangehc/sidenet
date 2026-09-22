"""Offline setup-script tests: installers are faked; only temporary venvs are created."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1]


class SetupEnvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="setup-script-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "repo with spaces"
        self.project.mkdir()
        shutil.copy2(SOURCE / "setup_env.sh", self.project / "setup_env.sh")
        for name in ("pyproject.toml", "requirements.txt"):
            if (SOURCE / name).exists():
                shutil.copy2(SOURCE / name, self.project / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.log = self.root / "installer.jsonl"
        self.env = dict(os.environ, PATH=f"{self.bin}:/usr/bin:/bin",
                        INSTALL_LOG=str(self.log), MOCK_CUDA="", CUDA_VISIBLE_DEVICES="0")
        self.write_command("nvidia-smi", 'printf "CUDA Version: %s\\n" "$MOCK_CUDA"\n')
        # Real venv creation is cheap and offline; dependency installation is never run.
        mock_uv = r"""
import json
import os
from pathlib import Path
import subprocess
import sys
args = sys.argv[1:]
with open(os.environ["INSTALL_LOG"], "a") as handle:
    handle.write(json.dumps(args) + "\n")
if args[0] == "venv":
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", args[-1]], check=True)
elif any(arg.startswith("torch==") for arg in args):
    sys.exit(43)  # Stop before the health check; this is a routing test.
"""
        self.write_command("uv", f"exec {shlex.quote(sys.executable)} -c {shlex.quote(mock_uv)} \"$@\"\n")

    def write_command(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_setup(self, *args):
        return subprocess.run(
            ["/bin/bash", str(self.project / "setup_env.sh"), *args],
            cwd=self.root, env=self.env, text=True, capture_output=True, timeout=30,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def create_venv(self):
        environment = self.project / ".venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
        return environment

    def test_help_needs_no_environment(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.project / ".venv").exists())
        self.assertEqual(self.calls(), [])

    def test_check_missing_environment_does_not_create_or_install(self):
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("虚拟环境不存在", result.stderr)
        self.assertFalse((self.project / ".venv").exists())
        self.assertEqual(self.calls(), [])

    def test_check_incomplete_dependencies_does_not_install(self):
        self.create_venv()
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("环境检查失败", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_invalid_environment_is_preserved(self):
        environment = self.project / ".venv"
        environment.mkdir()
        marker = environment / "keep.txt"
        marker.write_text("keep me")
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不是完整的虚拟环境", result.stderr)
        self.assertEqual(marker.read_text(), "keep me")
        self.assertEqual(self.calls(), [])

    def test_reject_interpreter_outside_target_environment(self):
        environment = self.project / ".venv"
        (environment / "bin").mkdir(parents=True)
        (environment / "pyvenv.cfg").write_text("placeholder")
        interpreter = environment / "bin/python"
        interpreter.write_text(f"#!/bin/bash\nexec {shlex.quote(sys.executable)} \"$@\"\n")
        interpreter.chmod(0o755)
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("解释器没有指向指定虚拟环境", result.stderr)
        self.assertEqual(self.calls(), [])

    def test_bad_arguments_fail_before_creation(self):
        for args in (("--venv",), ("--python",), ("--unknown",), ("--venv", "/")):
            with self.subTest(args=args):
                result = self.run_setup(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.project / ".venv").exists())
                self.assertEqual(self.calls(), [])

    def test_relative_environment_and_uv_install_target(self):
        result = self.run_setup("--venv", "custom env")
        self.assertNotEqual(result.returncode, 0)  # No real packages are installed.
        calls = self.calls()
        target = str(self.project / "custom env")
        self.assertEqual(calls[0], ["venv", "--python", "3.12", target])
        installs = [call for call in calls if call[:2] == ["pip", "install"]]
        self.assertTrue(installs)
        for call in installs:
            self.assertEqual(call[2:4], ["--python", target + "/bin/python"])

    def test_pip_fallback_is_isolated_and_targets_venv(self):
        environment = self.create_venv()
        (self.bin / "uv").unlink()
        site = next((environment / "lib").glob("python*/site-packages"))
        package = site / "pip"
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "__main__.py").write_text(
            'import json, os, sys\n'
            'if "--version" in sys.argv: sys.exit(0)\n'
            'with open(os.environ["INSTALL_LOG"], "a") as handle:\n'
            '    handle.write(json.dumps(sys.argv[1:]) + "\\n")\n'
            'sys.exit(43)\n'
        )
        result = self.run_setup()
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertEqual(self.calls()[0][:3], ["--isolated", "install", "--disable-pip-version-check"])

    def test_auto_selects_official_wheel_for_driver(self):
        for maximum, channel in (("", "cpu"), ("12.4", "cpu"), ("12.6", "cu126"),
                                 ("12.8", "cu128"), ("13.1", "cu128")):
            with self.subTest(maximum=maximum):
                self.env["MOCK_CUDA"] = maximum
                result = self.run_setup()
                self.assertEqual(result.returncode, 43, result.stderr)
                torch_call = self.calls()[-1]
                self.assertIn(f"torch==2.8.0+{channel}", torch_call)
                self.assertEqual(torch_call[-2:], ["--index-url", f"https://download.pytorch.org/whl/{channel}"])

    def test_cpu_override_and_repeat_do_not_recreate_environment(self):
        self.env["MOCK_CUDA"] = "13.1"
        for _ in range(2):
            result = self.run_setup("--device", "cpu")
            self.assertEqual(result.returncode, 43, result.stderr)
            self.assertIn("torch==2.8.0+cpu", self.calls()[-1])
        self.assertEqual(sum(call[0] == "venv" for call in self.calls()), 1)

    def test_explicit_cuda_channel(self):
        result = self.run_setup("--cuda", "cu126")
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertIn("torch==2.8.0+cu126", self.calls()[-1])

    def test_required_cuda_without_supported_driver_fails_before_install(self):
        result = self.run_setup("--device", "cuda")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未检测到支持 CUDA", result.stderr)
        self.assertFalse(any(call[:2] == ["pip", "install"] for call in self.calls()))

    def test_hidden_gpu_selects_cpu(self):
        self.env.update(MOCK_CUDA="13.1", CUDA_VISIBLE_DEVICES="-1")
        result = self.run_setup()
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertIn("torch==2.8.0+cpu", self.calls()[-1])

    def test_device_argument_conflicts(self):
        for args in (("--device", "bad"), ("--cuda", "cu999"),
                     ("--device", "cpu", "--cuda", "cu128")):
            with self.subTest(args=args):
                result = self.run_setup(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.project / ".venv").exists())
                self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
