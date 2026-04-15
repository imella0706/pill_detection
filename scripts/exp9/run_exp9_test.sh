#!/bin/bash
# Exp 9: Multi-Scale WBF Pipeline
# ---------------------------------------------------------
set -euo pipefail

MODEL_PATH="runs/pill_exp5_yolo11s_copypaste/weights/best.pt"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "=================================================="
echo "[Exp 9] Multi-scale WBF pipeline start"
echo " - model: ${MODEL_PATH}"
echo " - scales: 640, 960, 1024"
echo " - conf/iou: 0.20 / 0.60"
echo " - python: ${PYTHON_BIN}"

echo "=================================================="
echo "[1/4] 640px inference -> submission/exp9_submission_640.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 640 --conf 0.20 --iou 0.60 --output submission/exp9_submission_640.csv --no-save-config

echo "=================================================="
echo "[2/4] 960px inference -> submission/exp9_submission_960.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 960 --conf 0.20 --iou 0.60 --output submission/exp9_submission_960.csv --no-save-config

echo "=================================================="
echo "[3/4] 1024px inference -> submission/exp9_submission_1024.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp9_submission_1024.csv --no-save-config

echo "=================================================="
echo "[4/4] WBF ensemble -> submission/exp9_final_wbf.csv"
"${PYTHON_BIN}" src/ensemble_wbf.py \
    --csvs submission/exp9_submission_640.csv submission/exp9_submission_960.csv submission/exp9_submission_1024.csv \
    --output submission/exp9_final_wbf.csv \
    --iou 0.60

echo "=================================================="
echo "[Exp 9] Done -> submission/exp9_final_wbf.csv"
