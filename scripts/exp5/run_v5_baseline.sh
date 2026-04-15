#!/bin/bash
# scripts/onboard_baseline.sh
# Exp 5 베이스라인을 팀원 환경에 즉시 세팅하는 자동화 스크립트.

set -e # 에러 발생 시 즉시 중단 (안정성 확보)

echo "=================================================="
echo " 💊 [PillaTech] Team Onboarding Pipeline v1.0"
echo "=================================================="

# 1. 라이브러리 설치 (Conda 환경 활성화 상태 가정)
echo "[1/3] Installing verified dependencies from requirements.txt..."
pip install -r requirements.txt

# 2. 원본 데이터 유무 확인 (MLOps 방어 로직)
if [ ! -d "data/raw/sprint_ai_project1_data" ]; then
    echo "  Error: 원본 데이터셋을 찾을 수 없습니다."
    echo " data/raw/sprint_ai_project1_data/ 폴더에 이미지를 배치한 후 다시 실행하세요."
    exit 1
fi

# 3. 데이터 전처리 파이프라인 실행
echo "[2/3] Running preprocessing.py (Merged JSON & Split)..."
python preprocessing.py

echo "[3/3] Running prepare_yolo_dataset.py (YOLO Format Conversion)..."
python prepare_yolo_dataset.py

echo "=================================================="
echo " ✅ 모든 데이터 준비가 완료되었습니다!"
echo "--------------------------------------------------"
echo " 🚀 다음 명령어로 바로 학습을 시작할 수 있습니다:"
echo " python train_yolov8.py --config configs/exp5_train.yaml"
echo "=================================================="
