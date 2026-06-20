#!/bin/bash

# =========================================================
# Blue-Green Deployment Orchestrator (MVP Stage A)
# =========================================================
# 이 스크립트는 Nginx 설정 파일을 분석해 현재 활성 Blue/Green 컨테이너를 찾고,
# 반대편 컨테이너에 새 Gunicorn(ASGI worker) 워커를 띄운 뒤, 헬스체크를 통과하면
# Nginx upstream target을 전환하는 Blue-Green 배포 스크립트다.


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
    curl -sS --connect-timeout 2 --max-time 5 -X POST -H "Content-type: application/json" \
        --data "{\"text\":\"[$level] $msg\"}" \
        "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
}

# 1. 현재 Nginx 설정에서 활성화된 Blue/Green 컨테이너 식별
NGINX_CONF="deploy/nginx.conf"
BLUE_DEBUG_PORT="${BLUE_DEBUG_PORT:-8001}"
GREEN_DEBUG_PORT="${GREEN_DEBUG_PORT:-8002}"

debug_port_for_container() {
    case "$1" in
        blue) echo "$BLUE_DEBUG_PORT" ;;
        green) echo "$GREEN_DEBUG_PORT" ;;
        *) echo "" ;;
    esac
}

opposite_container() {
    case "$1" in
        blue) echo "green" ;;
        green) echo "blue" ;;
        *) echo "blue" ;;
    esac
}

CURRENT_BACKEND=$(grep -oP 'set \$api_backend "\K[^"]+' "$NGINX_CONF" | head -n 1)
CURRENT_CONTAINER="none"
if [[ "$CURRENT_BACKEND" == pill-api-blue:* ]]; then
    CURRENT_CONTAINER="blue"
elif [[ "$CURRENT_BACKEND" == pill-api-green:* ]]; then
    CURRENT_CONTAINER="green"
fi

if [ "$CURRENT_CONTAINER" = "none" ]; then
    NEW_CONTAINER="blue"
    OLD_CONTAINER="none"
else
    OLD_CONTAINER="$CURRENT_CONTAINER"
    NEW_CONTAINER="$(opposite_container "$CURRENT_CONTAINER")"
fi
NEW_DEBUG_PORT="$(debug_port_for_container "$NEW_CONTAINER")"
OLD_DEBUG_PORT="$(debug_port_for_container "$OLD_CONTAINER")"

echo "[CHECK] 현재 활성 컨테이너: $OLD_CONTAINER${OLD_DEBUG_PORT:+ debug_port=$OLD_DEBUG_PORT}"
echo "[INIT] 신규 컨테이너 기동 중: $NEW_CONTAINER debug_port=$NEW_DEBUG_PORT"
notify_slack "P3" "배포 시작: old=$OLD_CONTAINER${OLD_DEBUG_PORT:+(debug_port=$OLD_DEBUG_PORT)}, new=$NEW_CONTAINER(debug_port=$NEW_DEBUG_PORT), host=$(hostname)"

# Gunicorn runtime defaults (override 가능)
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-30}"
APP_MODULE="src.api.main:app"
DOCKER_NETWORK="${DOCKER_NETWORK:-pill_detection_default}"

# 2. 신규 파이프라인 (Docker 컨테이너) 빌드 및 기동
echo "[BUILD] 도커 이미지 빌드 중 (이전 빌드 캐시 활용)..."
docker build -t pill-api-image . > /dev/null

NEW_CONTAINER_NAME="pill-api-${NEW_CONTAINER}"
echo "[RUN] 신규 컨테이너($NEW_CONTAINER_NAME) 기동 중..."

# 충돌 방지를 위해 남아있는 잔해 제거
docker rm -f "$NEW_CONTAINER_NAME" 2>/dev/null || true

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
    --name "$NEW_CONTAINER_NAME" \
    --network "$DOCKER_NETWORK" \
    -p "127.0.0.1:$NEW_DEBUG_PORT:8000" \
    -v "$HOST_PROJECT_DIR:$HOST_PROJECT_DIR:rw" \
    "${DOCKER_ENV_ARGS[@]}" \
    pill-api-image

echo "[WAIT] 신규 컨테이너($NEW_CONTAINER_NAME) 기동 및 모델 웜업 대기 중..."

# 3. Health Check 루프 (최대 30초 대기)
# [Design Intent]
# - 호스트 포트 포워딩/WSL loopback 경로가 꼬이면, 컨테이너 내부 앱은 정상이더라도
#   host curl 검사에서 오류(503)가 발생할 수 있다.
# - readiness는 "신규 컨테이너 내부의 :8000 앱 상태"를 직접 검사해 판정한다.
MAX_RETRIES=30
HEALTHY=false

for i in $(seq 1 $MAX_RETRIES); do
    sleep 2
    if docker exec "$NEW_CONTAINER_NAME" python -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" \
        >/dev/null 2>&1; then
        STATUS_CODE="200"
    else
        STATUS_CODE="000"
    fi
    
    if [ "$STATUS_CODE" == "200" ]; then
        HEALTHY=true
        echo "[SUCCESS] 신규 컨테이너 준비 완료"
        break
    fi
    echo "  ...응답 대기 중 (상태 코드: $STATUS_CODE). 재시도 $i/$MAX_RETRIES"
done

# 4. 검증 실패 시 Rollback (신규 컨테이너 파기)
if [ "$HEALTHY" = false ]; then
    echo "[ERROR] 치명적 결함: 제한 시간 내 신규 컨테이너 기동 또는 모델 로드 실패"
    echo "[ROLLBACK] 롤백 수행 - 신규 컨테이너($NEW_CONTAINER_NAME) 강제 삭제"
    notify_slack "P1" "배포 중단: 신규 컨테이너 헬스체크 실패. 기존 컨테이너 유지 old=$OLD_CONTAINER${OLD_DEBUG_PORT:+(debug_port=$OLD_DEBUG_PORT)}, new=$NEW_CONTAINER(debug_port=$NEW_DEBUG_PORT), host=$(hostname)"
    echo "[DEBUG] 신규 컨테이너 로그 (최근 120줄):"
    docker logs --tail 120 "$NEW_CONTAINER_NAME" || true
    docker rm -f "$NEW_CONTAINER_NAME" > /dev/null || true
    exit 1
fi

# 5. 검증 통과 시 Nginx 스위칭 (Traffic Routing)
echo "[SWITCH] Nginx 업스트림 트래픽을 $NEW_CONTAINER 컨테이너로 전환 중..."
# Nginx conf 파일 안의 기존 target을 새 API 컨테이너로 치환
# (sed -i는 inode를 변경해 도커 마운트를 깨뜨리므로 cat으로 덮어쓰기 사용)
if grep -q 'set \$api_backend' "$NGINX_CONF"; then
    sed -E 's#set \$api_backend "[^"]+";#set $api_backend "pill-api-'"$NEW_CONTAINER"':8000";#g' "$NGINX_CONF" > temp_nginx.conf
else
    sed -E "s/server (host\.docker\.internal:[0-9]+|pill-api-(blue|green):8000);/server pill-api-$NEW_CONTAINER:8000;/g" "$NGINX_CONF" > temp_nginx.conf
fi
cat temp_nginx.conf > $NGINX_CONF
rm temp_nginx.conf

# 만약 Docker로 Nginx를 띄워두었다면 리로드 명령 전송
if docker ps | grep -q 'pill-nginx'; then
    docker exec pill-nginx nginx -s reload
    echo "[SUCCESS] Nginx 리로드 완료. 트래픽 전환 완료 ($NEW_CONTAINER debug_port=$NEW_DEBUG_PORT)"
    notify_slack "P2" "배포 성공: 트래픽 전환 완료 old=$OLD_CONTAINER${OLD_DEBUG_PORT:+(debug_port=$OLD_DEBUG_PORT)}, new=$NEW_CONTAINER(debug_port=$NEW_DEBUG_PORT), host=$(hostname)"
else
    echo "[WARN] 'pill-nginx' 컨테이너를 찾을 수 없음. Nginx 실행 상태 확인 필요"
    notify_slack "P2-WARN" "배포 경고: pill-nginx 미탐지로 리로드 미수행 old=$OLD_CONTAINER${OLD_DEBUG_PORT:+(debug_port=$OLD_DEBUG_PORT)}, new=$NEW_CONTAINER(debug_port=$NEW_DEBUG_PORT), host=$(hostname)"
fi

# 6. 구버전 컨테이너 보존
# [Design Intent]
# - Blue-Green 배포의 핵심은 새 컨테이너 전환 후 문제가 보이면 기존 컨테이너로 빠르게 되돌릴 수 있다는 점이다.
# - Stage A에서는 트래픽 전환 직후 구버전 컨테이너를 자동 삭제하지 않는다.
# - 운영자가 `/health`, `/predict`, 로그를 확인한 뒤 필요할 때 직접 삭제한다.
# - 자동삭제 정책이 필요해지면 아래와 같은 별도 cleanup 명령을 명시적으로 실행한다.
#   docker rm -f "pill-api-$OLD_CONTAINER"
if [ "$OLD_CONTAINER" != "none" ]; then
    echo "[KEEP] 구버전 컨테이너 유지: $OLD_CONTAINER${OLD_DEBUG_PORT:+ debug_port=$OLD_DEBUG_PORT}"
    echo "[KEEP] 검증 후 정리하려면 docker rm -f pill-api-$OLD_CONTAINER 를 직접 실행하세요."
fi

echo "[DONE] 무중단 배포 완료"
