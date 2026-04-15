#!/bin/bash
# scripts/exp10/run_exp10_ensemble_final.sh
# 이 스크립트는 프로젝트 루트(PillaTech_team04)에서 실행해야 합니다.
# 실행 예: bash scripts/exp10/run_exp10_ensemble_final.sh

# 1. 디렉토리 준비
mkdir -p submission/exp10_val
mkdir -p metrics

echo "=========================================================="
echo " [Exp 10] 3-Seed Ensemble & Evaluation Pipeline Start"
echo "=========================================================="

# 시드 목록
SEEDS=(42 123 777)

echo "=== [1/3] 개별 모델 추론 시작 (Test & Val) ==="
for SEED in "${SEEDS[@]}"; do
    WEIGHTS="runs/pill_exp10_seed${SEED}/weights/best.pt"
    
    if [ ! -f "$WEIGHTS" ]; then
        echo " Error: 가중치 파일을 찾을 수 없습니다: $WEIGHTS"
        exit 1
    fi

    echo "----------------------------------------------------------"
    echo " Processing Seed $SEED..."
    
    # A. 캐글 제출용 (Test Images)
    echo "   > Inferencing on TEST set..."
    python src/test_custom_v12.py --model "$WEIGHTS" \
        --imgsz 960 --conf 0.20 --iou 0.60 \
        --output submission/exp10_seed${SEED}_test.csv --no-save-config
    
    # B. 로컬 검증용 (Val Images)
    echo "   > Inferencing on VALIDATION set..."
    python src/test_custom_v12.py --model "$WEIGHTS" \
        --imgsz 960 --conf 0.20 --iou 0.60 \
        --test_images data/datasets/yolo/copy_paste/cp_t20_data_seed42/images/val \
        --output submission/exp10_val/exp10_seed${SEED}_val.csv --no-save-config
done

echo "=========================================================="
echo "=== [2/3] WBF 앙상블 실행 ( 병합 및 융합 ) ==="
echo "=========================================================="

# A. 캐글 제출용 병합
echo " Fusing TEST predictions..."
python src/ensemble_wbf.py --iou 0.6 \
    --csvs submission/exp10_seed42_test.csv submission/exp10_seed123_test.csv submission/exp10_seed777_test.csv \
    --output submission/exp10_final_submission.csv \
    --image_dir data/raw/test_images

# B. 로컬 검증용 병합
echo " Fusing VALIDATION predictions..."
python src/ensemble_wbf.py --iou 0.6 \
    --csvs submission/exp10_val/exp10_seed42_val.csv submission/exp10_val/exp10_seed123_val.csv submission/exp10_val/exp10_seed777_val.csv \
    --output submission/exp10_val/exp10_final_val_ensembled.csv \
    --image_dir data/datasets/yolo/copy_paste/cp_t20_data_seed42/images/val

echo "=========================================================="
echo "=== [3/3] 최종 앙상블 성능 평가 (mAP 산출) ==="
echo "=========================================================="

python src/eval_csv_map.py --pred_csv submission/exp10_val/exp10_final_val_ensembled.csv \
    --experiment exp10_ensemble --model_name yolo11s --iou 0.6 --save_json metrics/exp10_final_ensemble_metrics.json

echo "=========================================================="
echo " ALL TASKS COMPLETED!"
echo " Final Submission: submission/exp10_final_submission.csv"
echo " Validation Metrics: metrics/exp10_final_ensemble_metrics.json"
echo "=========================================================="
