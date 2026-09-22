#!/usr/bin/env bash
# Linux / WSL. This file is independent of the sibling repository.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_REQUEST="3.12"
CHECK_ONLY=0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

die() { printf '错误：%s\n' "$*" >&2; exit 1; }
log() { printf '[环境] %s\n' "$*"; }
require_value() {
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 需要一个参数"
}

DEVICE="auto"
CUDA_CHANNEL=""
usage() {
  cat <<'HELP'
用法：bash setup_env.sh [选项]
  默认              创建/复用本项目 .venv，安装依赖并验证运算。
  --check           只检查现有环境，不安装或更新依赖；失败返回非零。
  --device auto     自动检测 NVIDIA 驱动；选 cu128/cu126，否则用 CPU。
  --device cpu      安装 CPU 版 PyTorch。
  --device cuda     要求 CUDA；无法检测合适驱动时直接报错。
  --cuda cu126|cu128 显式选择 CUDA wheel，同时要求 CUDA 可用。
  --python 版本/路径 新建环境的 Python，默认 3.12；支持 3.12/3.13。
  --venv 路径       环境路径，相对路径以本项目为基准，默认 .venv。
  -h, --help        显示帮助。

依赖：requirements.txt + PyTorch 2.8.0 官方 CPU/CUDA wheel。
脚本不安装 NVIDIA 驱动或系统 CUDA Toolkit。安装后验证实际张量运算。
HELP
}
while (( $# )); do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --device) require_value "$@"; DEVICE="$2"; shift 2 ;;
    --cuda) require_value "$@"; CUDA_CHANNEL="$2"; shift 2 ;;
    --python) require_value "$@"; PYTHON_REQUEST="$2"; shift 2 ;;
    --venv) require_value "$@"; VENV_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1（使用 --help 查看用法）" ;;
  esac
done
case "$DEVICE" in auto|cpu|cuda) ;; *) die "--device 只能是 auto、cpu 或 cuda" ;; esac
case "$CUDA_CHANNEL" in ""|cu126|cu128) ;; *) die "--cuda 只能是 cu126 或 cu128" ;; esac
if [[ -n "$CUDA_CHANNEL" ]]; then
  [[ "$DEVICE" != cpu ]] || die "--device cpu 不能与 --cuda 同时使用"
  DEVICE="cuda"
fi

# Resolve relative environment paths against this repository, not the caller's cwd.
[[ "$VENV_DIR" = /* ]] || VENV_DIR="$PROJECT_DIR/$VENV_DIR"
VENV_DIR="${VENV_DIR%/}"
[[ -n "$VENV_DIR" ]] || die "不能使用文件系统根目录作为虚拟环境"
ENV_PY="$VENV_DIR/bin/python"

python_supported() {
  "$1" -I -B -c 'import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] <= (3, 13)))' >/dev/null 2>&1
}

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
  [[ -f "$VENV_DIR/pyvenv.cfg" && -x "$ENV_PY" ]] ||
    die "$VENV_DIR 不是完整的虚拟环境。请用 --venv 指定新目录；脚本不会删除旧目录。"
else
  (( CHECK_ONLY == 0 )) || die "虚拟环境不存在：$VENV_DIR；去掉 --check 可创建。"
  if command -v uv >/dev/null 2>&1; then
    log "使用 uv 创建环境（必要时下载 Python）：$VENV_DIR"
    uv venv --python "$PYTHON_REQUEST" "$VENV_DIR"
  else
    PYTHON_BIN=""
    for candidate in "$PYTHON_REQUEST" "python$PYTHON_REQUEST"; do
      if command -v "$candidate" >/dev/null 2>&1 && python_supported "$candidate"; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
    # Only the default request may fall back to another supported local Python.
    if [[ -z "$PYTHON_BIN" && "$PYTHON_REQUEST" == "3.12" ]]; then
      for candidate in python3.12 python3.13 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && python_supported "$candidate"; then
          PYTHON_BIN="$candidate"
          break
        fi
      done
    fi
    [[ -n "$PYTHON_BIN" ]] ||
      die "找不到 Python 3.12/3.13 或 uv。请安装 uv（https://docs.astral.sh/uv/getting-started/installation/），或用 --python 指定解释器。"
    log "使用 $PYTHON_BIN 创建环境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR" ||
      die "创建失败；Debian/Ubuntu 请检查对应的 python3.x-venv 包，然后使用新的 --venv 目录重试。"
  fi
fi

python_supported "$ENV_PY" || die "环境需要 Python 3.12 或 3.13；请用 --venv 指定新目录。"
"$ENV_PY" -I -B - "$VENV_DIR" <<'PY'
import pathlib
import sys
expected = pathlib.Path(sys.argv[1]).resolve()
if sys.prefix == sys.base_prefix or pathlib.Path(sys.prefix).resolve() != expected:
    sys.exit("错误：解释器没有指向指定虚拟环境，停止安装。")
print(f"[环境] Python {sys.version.split()[0]}: {sys.executable}")
PY

install_packages() {
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ENV_PY" "$@"
  else
    if ! "$ENV_PY" -I -B -m pip --version >/dev/null 2>&1; then
      "$ENV_PY" -I -B -m ensurepip --upgrade ||
        die "环境缺少 pip/ensurepip，请安装 uv 后重试。"
    fi
    "$ENV_PY" -I -B -m pip --isolated install --disable-pip-version-check "$@"
  fi
}

check_dependencies() {
  if command -v uv >/dev/null 2>&1; then
    uv pip check --python "$ENV_PY"
  elif "$ENV_PY" -I -B -m pip --version >/dev/null 2>&1; then
    "$ENV_PY" -I -B -m pip --isolated check
  fi
}

if [[ "$DEVICE" == cpu ]]; then
  CHANNEL="cpu"
elif [[ -n "$CUDA_CHANNEL" ]]; then
  CHANNEL="$CUDA_CHANNEL"
else
  CHANNEL="$("$ENV_PY" -I -B - <<'PY'
import os
import re
import subprocess

channel = "cpu"
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
    # Preserve a working supported CUDA build, including systems without nvidia-smi.
    try:
        import torch
        if torch.__version__.split("+")[0] == "2.8.0" and torch.cuda.is_available():
            if torch.version.cuda in ("12.6", "12.8"):
                channel = "cu" + torch.version.cuda.replace(".", "")
    except Exception:
        pass
    if channel == "cpu":
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
            if result.returncode == 0 and match:
                maximum = tuple(map(int, match.groups()))
                if maximum >= (12, 8):
                    channel = "cu128"
                elif maximum >= (12, 6):
                    channel = "cu126"
        except (OSError, subprocess.TimeoutExpired):
            pass
print(channel)
PY
)"
  if [[ "$DEVICE" == cuda && "$CHANNEL" == cpu ]]; then
    die "未检测到支持 CUDA 12.6+ 的可见 NVIDIA 驱动。请检查 nvidia-smi / CUDA_VISIBLE_DEVICES，或使用 --device cpu。"
  fi
fi
log "PyTorch 目标：2.8.0+$CHANNEL（选择方式：$DEVICE）"
if [[ "$DEVICE" == auto && "$CHANNEL" == cpu ]]; then
  log "未检测到支持 cu126/cu128 的可见 GPU，使用 CPU；可通过 --device cuda 要求 GPU。"
fi

if (( CHECK_ONLY == 0 )); then
  install_packages -r "$PROJECT_DIR/requirements.txt"
  # Include the local version suffix so switching CPU/CUDA actually replaces torch.
  install_packages "torch==2.8.0+$CHANNEL" --index-url "https://download.pytorch.org/whl/$CHANNEL"
fi

"$ENV_PY" -I -B - "$PROJECT_DIR" "$CHANNEL" <<'PY'
import importlib
import importlib.metadata
from pathlib import Path
import re
import sys

root, channel = Path(sys.argv[1]), sys.argv[2]
try:
    for requirement in (root / "requirements.txt").read_text().splitlines():
        requirement = requirement.strip()
        if not requirement or requirement.startswith("#"):
            continue
        match = re.fullmatch(r"([\w-]+)>=(\d+(?:\.\d+)*),<(\d+)", requirement)
        if not match:
            raise RuntimeError(f"检查器需要更新以支持依赖表达式：{requirement}")
        name, minimum, upper = match.groups()
        actual = importlib.metadata.version(name)
        release = actual.split("+")[0]
        if not re.fullmatch(r"\d+(?:\.\d+)*", release):
            raise RuntimeError(f"{name} {actual} 不是稳定发行版")
        version = tuple(map(int, release.split(".")))
        if not (tuple(map(int, minimum.split("."))) <= version < (int(upper),)):
            raise RuntimeError(f"{name} {actual} 不满足 {requirement}")
        print(f"[依赖] {name} {actual}")
    import numpy as np
    import torch
    import yaml
    expected_cuda = {"cpu": None, "cu126": "12.6", "cu128": "12.8"}[channel]
    if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != expected_cuda:
        raise RuntimeError(f"PyTorch {torch.__version__} / CUDA {torch.version.cuda} 与目标 2.8.0+{channel} 不符")
    print(f"[依赖] torch {torch.__version__}")
    device = "cpu" if channel == "cpu" else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA wheel 已安装，但 GPU 不可用。检查驱动、GPU 可见性；或重新运行 --device cpu")
    tensor = torch.from_numpy(np.ones((4, 4), dtype=np.float32)).to(device).requires_grad_()
    (tensor @ tensor).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    sys.path.insert(0, str(root))
    for module in ("sidenet", "sidenet_data", "train_sidenet", "infer", "prepare_mixed_split"):
        importlib.import_module(module)
except Exception as exc:
    sys.exit(f"环境检查失败：{exc}\n请运行不带 --check 的 setup_env.sh，或调整 --device/--cuda。")
print(f"[通过] {device} 前向/反向运算、NumPy 互通和训练模块导入。")
PY
check_dependencies
printf '\n完成。训练示例（请先生成数据并划分）：\n  cd %q\n' "$PROJECT_DIR"
if [[ "$CHANNEL" == cpu ]]; then
  printf '  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 %q train_sidenet.py configs/dgcnn_mixed.yaml --device cpu\n' "$ENV_PY"
else
  printf '  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 %q train_sidenet.py configs/dgcnn_mixed.yaml --device cuda\n' "$ENV_PY"
fi
