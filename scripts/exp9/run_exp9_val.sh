#!/bin/bash
# Exp 9: Multi-Scale WBF validation pipeline
set -euo pipefail

MODEL_PATH="runs/pill_exp5_yolo11s_copypaste/weights/best.pt"
PYTHON_BIN="${PYTHON_BIN:-/home/imella0707/miniconda3/envs/codeit/bin/python}"
VAL_IMAGE_DIR="data/datasets/yolo/copy_paste/cp_t20_data_seed42/images/val"
VAL_LABEL_DIR="data/datasets/yolo/copy_paste/cp_t20_data_seed42/labels/val"

echo "=================================================="
echo "[Exp 9 Val] Multi-scale WBF validation pipeline start"
echo " - model: ${MODEL_PATH}"
echo " - scales: 640, 960, 1024"
echo " - conf/iou: 0.20 / 0.60"
echo " - image_dir: ${VAL_IMAGE_DIR}"
echo " - python: ${PYTHON_BIN}"

echo "=================================================="
echo "[1/5] 640px val inference -> submission/exp9_val_640.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 640 --conf 0.20 --iou 0.60 --output submission/exp9_val_640.csv --test_images "$VAL_IMAGE_DIR" --no-save-config

echo "=================================================="
echo "[2/5] 960px val inference -> submission/exp9_val_960.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 960 --conf 0.20 --iou 0.60 --output submission/exp9_val_960.csv --test_images "$VAL_IMAGE_DIR" --no-save-config

echo "=================================================="
echo "[3/5] 1024px val inference -> submission/exp9_val_1024.csv"
"${PYTHON_BIN}" src/test_custom_v12.py --model "$MODEL_PATH" --imgsz 1024 --conf 0.20 --iou 0.60 --output submission/exp9_val_1024.csv --test_images "$VAL_IMAGE_DIR" --no-save-config

echo "=================================================="
echo "[4/5] WBF ensemble -> submission/exp9_val_final_wbf.csv"
"${PYTHON_BIN}" src/ensemble_wbf.py \
    --csvs submission/exp9_val_640.csv submission/exp9_val_960.csv submission/exp9_val_1024.csv \
    --output submission/exp9_val_final_wbf.csv \
    --iou 0.60 \
    --image_dir "$VAL_IMAGE_DIR"

echo "=================================================="
echo "[5/5] Evaluate WBF CSV on val labels"
"${PYTHON_BIN}" src/eval_csv_map.py \
    --pred_csv submission/exp9_val_final_wbf.csv \
    --image_dir "$VAL_IMAGE_DIR" \
    --label_dir "$VAL_LABEL_DIR" \
    --experiment exp9 \
    --model_name yolo11s \
    --model_path "$MODEL_PATH" \
    --dataset_split val \
    --conf 0.20 \
    --iou 0.60 \
    --scales 640,960,1024 \
    --save_json metrics/exp9_val_metrics.json

echo "=================================================="
echo "[Exp 9 Val] Done"
