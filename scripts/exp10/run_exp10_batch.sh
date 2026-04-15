#!/usr/bin/env bash
# Exp 10: 3-Seed Training Sequential Batch Script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"

CONFIGS=(
  "configs/train/exp10_seed42_train.yaml"
  "configs/train/exp10_seed123_train.yaml"
  "configs/train/exp10_seed777_train.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "${cfg}" ]]; then
    echo "Error: config 파일을 찾을 수 없습니다: ${cfg}"
    exit 1
  fi
done

echo "=== STARTING EXP 10-1 (Seed 42) ==="
"${PYTHON_BIN}" train_yolo.py --config "configs/train/exp10_seed42_train.yaml"

echo "=== STARTING EXP 10-2 (Seed 123) ==="
"${PYTHON_BIN}" train_yolo.py --config "configs/train/exp10_seed123_train.yaml"

echo "=== STARTING EXP 10-3 (Seed 777) ==="
"${PYTHON_BIN}" train_yolo.py --config "configs/train/exp10_seed777_train.yaml"

echo "=== ALL EXP 10 JOBS FINISHED! ==="
