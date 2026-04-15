import os
import io
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# Global Variable: 모델 가중치 (Startup 시 1회 메모리 로드)
model = None

# 환경 변수로 모델 경로를 동적으로 주입 (기본값 설정)
# (Stage A: 로컬 .pt 파일 / Stage B: MLflow Registry URI)
MODEL_PATH = os.getenv("MODEL_PATH", "runs/exp15_train_baseline_yolo11s_2.0/weights/best.pt")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 생명주기 관리 (Rule 5-1: Load Once, Reuse Many)
    - 매 요청마다 모델을 로드하면 I/O 병목이 터지므로, Gunicorn/Uvicorn 워커 기동 시 1번만 모델을 램(RAM)에 올립니다.
    """
    global model
    try:
        print(f"[Init] Loading YOLO model from {MODEL_PATH}...")
        start_time = time.time()
        model = YOLO(MODEL_PATH)
        print(f"[Init] Model loaded successfully in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        model = None
    
    yield  # 이 지점에서 FastAPI 애플리케이션 시작
    
    # 종료 로직 (있을 경우)
    model = None
    print("[Shutdown] Model unloaded and resources released.")

app = FastAPI(
    title="Pill Detection Zero-Downtime API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """
    Blue-Green 배포 스위치 동작용 L7 헬스체크 엔드포인트.
    Nginx나 Load Balancer가 워커가 트래픽을 받을 준비가 됐는지 검사합니다.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Worker is unready.")
    return {"status": "ok", "message": "Ready to serve traffic."}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    단일 이미지 객체 탐지 추론 파이프라인.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")
    
    # 1. 메모리상에서 이미지 로드 (디스크 I/O 최소화)
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
    
    # 2. 모델 인퍼런스 (기본값: conf=0.20, iou=0.60)
    try:
        start_time = time.time()
        results = model(image, conf=0.20, iou=0.60)
        inference_time = time.time() - start_time
        
        # 3. 결과 파싱 및 JSON화
        predictions = []
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].tolist()
                c = box.cls.item()
                conf = box.conf.item()
                predictions.append({
                    "bbox": [round(x, 2) for x in b],
                    "class": int(c),
                    "confidence": round(conf, 4)
                })
        
        return JSONResponse(content={
            "inference_time_seconds": round(inference_time, 4),
            "predictions": predictions
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {e}")
