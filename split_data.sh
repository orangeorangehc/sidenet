#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '找不到 Python 环境：%s\n请先运行 bash %s/setup_env.sh，或设置 PYTHON_BIN 为解释器的绝对路径。\n' "$PYTHON_BIN" "$PROJECT_DIR" >&2
  exit 1
fi
cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -u workflow.py split --config configs/split_data.yaml "$@"
