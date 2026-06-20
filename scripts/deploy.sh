#!/bin/bash

# =========================================================
# Blue-Green Deployment Orchestrator (MVP Stage A)
# =========================================================
# 이 스크립트는 Nginx 설정 파일을 분석해 현재 띄워진 포트(Blue)를 찾고,
# 반대편 포트(Green)에 새 Gunicorn(ASGI worker) 워커를 띄운 뒤, 헬스체크를 통과하면
# Nginx 방향을 틀어버리는 "Zero-Downtime(무중단)" 스위칭 스크립트다.


echo "[INFO] Blue-Green 배포 시작..."

# Optional: load env vars from project `.env` (do not commit secrets).
# This keeps the deploy script free of hardcoded credentials.
ENV_HAS_MODEL_URI=false
if [ -f ".env" ]; then
    if grep -Eq '^[[:space:]]*MODEL_URI=' ".env"; then
        ENV_HAS_MODEL_URI=true
    fi
    set -a
    # shellcheck disable=SC1091
    . ".env"
    set +a
fi

# MODEL_URI selection order (deterministic):
# 1) DEPLOY_MODEL_URI (one-shot override at command line)
# 2) MODEL_URI defined in .env
# 3) default production alias
DEFAULT_MODEL_URI="models:/pill_detection_2026@production"
if [ -n "${DEPLOY_MODEL_URI:-}" ]; then
    MODEL_URI="$DEPLOY_MODEL_URI"
elif [ "$ENV_HAS_MODEL_URI" = true ] && [ -n "${MODEL_URI:-}" ]; then
    MODEL_URI="$MODEL_URI"
else
    MODEL_URI="$DEFAULT_MODEL_URI"
fi
echo "[CFG] MODEL_URI=$MODEL_URI"

# Slack webhook (optional): set via env/secret, never hardcode in script.
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
notify_slack() {
    local level="$1"
    local msg="$2"
    [ -z "$SLACK_WEBHOOK_URL" ] && return 0
    curl -sS -X POST -H "Content-type: application/json" \
        --data "{\"text\":\"[$level] $msg\"}" \
        "$SLACK_WEBHOOK_URL" >/dev/null || true
}

# 1. 현재 Nginx 설정에서 활성화된 포트 식별
NGINX_CONF="deploy/nginx.conf"
CURRENT_PORT=$(grep -oP 'server host.docker.internal:\K\d+' $NGINX_CONF)

if [ "$CURRENT_PORT" == "8001" ]; then
    NEW_PORT="8002"
    OLD_PORT="8001"
elif [ "$CURRENT_PORT" == "8002" ]; then
    NEW_PORT="8001"
    OLD_PORT="8002"
else
    # default fallback
    NEW_PORT="8001"
    OLD_PORT="none"
fi

echo "[CHECK] 현재 활성 포트(Blue): $OLD_PORT"
echo "[INIT] 신규 워커(Green) 기동 중 - 포트: $NEW_PORT"
notify_slack "P3" "배포 시작: old=$OLD_PORT new=$NEW_PORT host=$(hostname)"

# Gunicorn runtime defaults (override 가능)
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-30}"
APP_MODULE="src.api.main:app"

# 2. 신규 파이프라인 (Docker 컨테이너) 빌드 및 기동
echo "[BUILD] 도커 이미지 빌드 중 (이전 빌드 캐시 활용)..."
docker build -t pill-api-image . > /dev/null

CONTAINER_NAME="pill-api-${NEW_PORT}"
echo "[RUN] 신규 컨테이너($CONTAINER_NAME) 기동 중..."

# 충돌 방지를 위해 남아있는 잔해 제거
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# 모델 레지스트리 접근 및 추론 성능 격리를 위해 Docker Run 수행
# [Design Intent]
# - 로컬 개발 환경(file 기반 MLflow store)에서는 모델 artifact 경로가 "호스트 절대경로"로 저장되는 경우가 있다.
# - 컨테이너 안에서 해당 경로를 그대로 접근 가능하게 하려면, 호스트 프로젝트 디렉터리를 "동일한 절대경로"로 마운트해야 한다.
# - 클라우드 환경에서는 GCS/S3 같은 원격 artifact store로 전환하는 게 정석이다.
HOST_PROJECT_DIR="$(pwd)"
TRACKING_URI="file:$HOST_PROJECT_DIR/mlruns"
EFFECTIVE_MODEL_URI="$MODEL_URI"
if [[ -n "$MODEL_URI" && "$MODEL_URI" == models:/* ]]; then
    echo "[RESOLVE] Registry URI -> local checkpoint path"
    if ! EFFECTIVE_MODEL_URI="$(python scripts/resolve_registry_model_path.py --model-uri "$MODEL_URI" --tracking-uri "$TRACKING_URI")"; then
        echo "[ERROR] Registry URI resolution failed: $MODEL_URI"
        notify_slack "P1" "배포 중단: 모델 URI 해석 실패 uri=$MODEL_URI host=$(hostname)"
        exit 1
    fi
    echo "[RESOLVE] resolved MODEL_URI=$EFFECTIVE_MODEL_URI"
fi

DOCKER_ENV_ARGS=(-e "MLFLOW_TRACKING_URI=$TRACKING_URI")
if [ -n "$EFFECTIVE_MODEL_URI" ]; then
    DOCKER_ENV_ARGS+=(-e "MODEL_URI=$EFFECTIVE_MODEL_URI")
fi
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$NEW_PORT:8000" \
    -v "$HOST_PROJECT_DIR:$HOST_PROJECT_DIR:rw" \
    "${DOCKER_ENV_ARGS[@]}" \
    pill-api-image

echo "[WAIT] Green 워커($CONTAINER_NAME) 기동 및 모델 웜업 대기 중..."

# 3. Health Check 루프 (최대 30초 대기)
# [Design Intent]
# - 호스트 포트 포워딩/WSL loopback 경로가 꼬이면, 컨테이너 내부 앱은 정상이더라도
#   host curl 검사에서 오탐(503)이 발생할 수 있다.
# - readiness는 "신규 Green 컨테이너 내부의 :8000 앱 상태"를 직접 검사해 판정한다.
MAX_RETRIES=30
HEALTHY=false

for i in $(seq 1 $MAX_RETRIES); do
    sleep 2
    if docker exec "$CONTAINER_NAME" python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" \
        >/dev/null 2>&1; then
        STATUS_CODE="200"
    else
        STATUS_CODE="000"
    fi
    
    if [ "$STATUS_CODE" == "200" ]; then
        HEALTHY=true
        echo "[SUCCESS] Green 워커 준비 완료"
        break
    fi
    echo "  ...응답 대기 중 (상태 코드: $STATUS_CODE). 재시도 $i/$MAX_RETRIES"
done

# 4. 검증 실패 시 Rollback (Green 컨테이너 파기)
if [ "$HEALTHY" = false ]; then
    echo "[ERROR] 치명적 결함: 제한 시간 내 Green 워커 기동 또는 모델 로드 실패"
    echo "[ROLLBACK] 롤백 수행 - Green 컨테이너($CONTAINER_NAME) 강제 삭제"
    notify_slack "P1" "배포 중단: Green 헬스체크 실패. 기존 Blue 유지 old=$OLD_PORT new=$NEW_PORT host=$(hostname)"
    echo "[DEBUG] Green 컨테이너 로그 (최근 120줄):"
    docker logs --tail 120 "$CONTAINER_NAME" || true
    docker rm -f "$CONTAINER_NAME" > /dev/null || true
    exit 1
fi

# 5. 검증 통과 시 Nginx 스위칭 (Traffic Routing)
echo "[SWITCH] Nginx 업스트림 트래픽을 포트 $NEW_PORT로 전환 중..."
# Nginx conf 파일 안의 8001 또는 8002 텍스트 치환
# (sed -i는 inode를 변경해 도커 마운트를 깨뜨리므로 cat으로 덮어쓰기 사용)
sed "s/server host.docker.internal:$OLD_PORT;/server host.docker.internal:$NEW_PORT;/g" $NGINX_CONF > temp_nginx.conf
cat temp_nginx.conf > $NGINX_CONF
rm temp_nginx.conf

# 만약 Docker로 Nginx를 띄워두었다면 리로드 명령 전송
if docker ps | grep -q 'pill-nginx'; then
    docker exec pill-nginx nginx -s reload
    echo "[SUCCESS] Nginx 리로드 완료. 트래픽 전환 완료 ($NEW_PORT)"
    notify_slack "P2" "배포 성공: 트래픽 전환 완료 old=$OLD_PORT new=$NEW_PORT host=$(hostname)"
else
    echo "[WARN] 'pill-nginx' 컨테이너를 찾을 수 없음. Nginx 실행 상태 확인 필요"
    notify_slack "P2-WARN" "배포 경고: pill-nginx 미탐지로 리로드 미수행 old=$OLD_PORT new=$NEW_PORT host=$(hostname)"
fi

# 6. 구형버전 (Blue) 컨테이너 종료 및 삭제
if [ "$OLD_PORT" != "none" ]; then
    echo "[CLEAN] 구버전(Blue) 컨테이너 종료 처리 중 - 포트: $OLD_PORT..."
    OLD_CONTAINER="pill-api-$OLD_PORT"
    if docker ps -a | grep -q "$OLD_CONTAINER"; then
        docker rm -f "$OLD_CONTAINER" > /dev/null
        echo "[SUCCESS] 구버전(Blue) 컨테이너($OLD_CONTAINER) 정상 삭제 완료"
    else
        echo "[WARN] 구버전 컨테이너($OLD_CONTAINER)를 찾을 수 없으나 진행에 문제가 없어 무시합니다."
    fi
fi

echo "[DONE] 무중단 배포 완료"
