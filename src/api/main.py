import os
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse, unquote
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image

# 환경 변수로 모델 URI를 동적으로 주입 (기본값: MLflow Registry production alias)
MODEL_URI = os.getenv("MODEL_URI", "models:/pill_detection_2026@production")


def resolve_model_path(model_uri: str) -> str:
    if not model_uri.startswith("models:/"):
        return model_uri

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        raise RuntimeError(
            "MODEL_URI is an MLflow registry URI but `mlflow` is unavailable."
        ) from exc

    model_name, alias_or_version, is_alias = parse_models_uri(model_uri)
    client = MlflowClient()
    if is_alias:
        mv = client.get_model_version_by_alias(model_name, alias_or_version)
    else:
        mv = client.get_model_version(model_name, alias_or_version)

    source_uri = getattr(mv, "source", None)
    if not source_uri:
        raise RuntimeError(f"source URI not found for model URI: {model_uri}")

    local_source_path = file_uri_to_path(source_uri)
    if local_source_path is not None:
        return str(select_model_checkpoint(local_source_path))

    downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=source_uri))
    return str(select_model_checkpoint(downloaded_path))


def parse_models_uri(model_uri: str) -> tuple[str, str, bool]:
    # models:/<name>@<alias> or models:/<name>/<version>
    remainder = model_uri[len("models:/") :].strip("/")
    if "@" in remainder:
        model_name, alias = remainder.rsplit("@", 1)
        if not model_name or not alias:
            raise ValueError(f"invalid registry alias URI: {model_uri}")
        return model_name, alias, True
    if "/" in remainder:
        model_name, version = remainder.rsplit("/", 1)
        if not model_name or not version:
            raise ValueError(f"invalid registry version URI: {model_uri}")
        return model_name, version, False
    raise ValueError(f"unsupported model URI format: {model_uri}")


def file_uri_to_path(uri: str) -> Path | None:
    if not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return Path(unquote(parsed.path))


def select_model_checkpoint(downloaded_path: Path) -> Path:
    if downloaded_path.is_file():
        return downloaded_path

    common_candidates = [downloaded_path / "best.pt", downloaded_path / "model.pt"]
    for candidate in common_candidates:
        if candidate.exists():
            return candidate

    # Registry artifact may point to an MLflow model package that does not include
    # a YOLO-native .pt checkpoint at the root. In this project we keep the raw
    # checkpoint under sibling artifact path: model_raw/best.pt.
    sibling_raw_best = downloaded_path.parent / "model_raw" / "best.pt"
    if sibling_raw_best.exists():
        return sibling_raw_best

    pt_files = sorted(downloaded_path.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found under: {downloaded_path}")

    for candidate in pt_files:
        if candidate.name == "best.pt":
            return candidate
    return pt_files[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버 생명주기 관리.
    - 모델 로드는 startup 시 1회 수행
    - 로드 실패 시 fail-fast로 worker startup을 중단
    """
    app.state.model = None
    app.state.model_init_error = None
    app.state.model_uri = MODEL_URI
    app.state.resolved_model_path = None
    app.state.worker_pid = os.getpid()
    app.state.app_instance_id = id(app)
    try:
        print(f"[Init] PID: {app.state.worker_pid} - Resolving model URI: {MODEL_URI}")
        start_time = time.time()
        resolved_model_path = resolve_model_path(MODEL_URI)
        app.state.resolved_model_path = str(resolved_model_path)
        print(f"[Init] Loading YOLO model from {resolved_model_path}...")
        app.state.model = YOLO(resolved_model_path)
        app.state.model_id = id(app.state.model)
        print(
            f"[Init] Model loaded successfully (ID: {app.state.model_id}) "
            f"in {time.time() - start_time:.2f}s"
        )
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        app.state.model_init_error = str(e)
        raise RuntimeError(f"Startup failed: {e}") from e
    
    yield  # 이 지점에서 FastAPI 애플리케이션 시작
    
    # 종료 로직 (있을 경우)
    app.state.model = None
    app.state.model_init_error = "model lifecycle shutdown"
    print(f"[Shutdown] PID: {app.state.worker_pid} - Model unloaded and resources released.")

app = FastAPI(
    title="Pill Detection Zero-Downtime API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check(request: Request):
    """
    Blue-Green 배포 스위치 동작용 L7 헬스체크 엔드포인트.
    Nginx나 Load Balancer가 워커가 트래픽을 받을 준비가 됐는지 검사합니다.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        error_detail = getattr(request.app.state, "model_init_error", None) or "Model is not loaded. Worker is unready."
        raise HTTPException(
            status_code=503,
            detail={
                "error": error_detail,
                "pid": os.getpid(),
                "app_instance_id": id(request.app),
                "startup_app_instance_id": getattr(request.app.state, "app_instance_id", None),
                "has_model_attr": hasattr(request.app.state, "model"),
                "model_is_none": model is None,
                "model_id": getattr(request.app.state, "model_id", None),
            },
        )
    return {
        "status": "ok",
        "message": "Ready to serve traffic.",
        "pid": os.getpid(),
        "app_instance_id": id(request.app),
        "startup_app_instance_id": getattr(request.app.state, "app_instance_id", None),
        "model_id": getattr(request.app.state, "model_id", None),
        "model_uri": getattr(request.app.state, "model_uri", MODEL_URI),
        "resolved_model_path": getattr(request.app.state, "resolved_model_path", None),
    }

@app.post("/predict")
async def predict_image(request: Request, file: UploadFile = File(...)):
    """
    단일 이미지 객체 탐지 추론 파이프라인.
    """
    model = getattr(request.app.state, "model", None)
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
