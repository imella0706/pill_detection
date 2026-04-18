#!/bin/bash

# =========================================================
# Blue-Green Deployment Orchestrator (MVP Stage A)
# =========================================================
# 이 스크립트는 Nginx 설정 파일을 분석해 현재 띄워진 포트(Blue)를 찾고,
# 반대편 포트(Green)에 새 Gunicorn(ASGI worker) 워커를 띄운 뒤, 헬스체크를 통과하면
# Nginx 방향을 틀어버리는 "Zero-Downtime(무중단)" 스위칭 스크립트다.

echo "[INFO] Starting Blue-Green Deployment..."

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

echo "[CHECK] Current Active Port (Blue): $OLD_PORT"
echo "[INIT] Spawning New Worker (Green) on Port: $NEW_PORT"

# Gunicorn runtime defaults (override 가능)
GUNICORN_WORKERS="${GUNICORN_WORKERS:-1}"
GUNICORN_TIMEOUT="${GUNICORN_TIMEOUT:-120}"
GUNICORN_GRACEFUL_TIMEOUT="${GUNICORN_GRACEFUL_TIMEOUT:-30}"
GUNICORN_KEEPALIVE="${GUNICORN_KEEPALIVE:-5}"
APP_MODULE="src.api.main:app"

# 2. 새로운 Gunicorn(FastAPI ASGI worker) 기동 (백그라운드)
# 핵심: FastAPI는 ASGI이므로 sync worker가 아닌 UvicornWorker를 강제한다.
conda run -n codeit gunicorn "$APP_MODULE" \
    -k uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:$NEW_PORT" \
    --workers "$GUNICORN_WORKERS" \
    --timeout "$GUNICORN_TIMEOUT" \
    --graceful-timeout "$GUNICORN_GRACEFUL_TIMEOUT" \
    --keep-alive "$GUNICORN_KEEPALIVE" &
NEW_PID=$!

echo "[WAIT] Waiting for Green Worker (PID: $NEW_PID) to warm up and load the YOLO model..."

# 3. Health Check 루프 (최대 30초 대기)
MAX_RETRIES=15
HEALTHY=false

for i in $(seq 1 $MAX_RETRIES); do
    sleep 2
    # health API 찌르기 (http 상태코드 반환)
    STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$NEW_PORT/health)
    
    if [ "$STATUS_CODE" == "200" ]; then
        HEALTHY=true
        echo "[SUCCESS] Green Worker is READY!"
        break
    fi
    echo "  ...still waiting (Status: $STATUS_CODE). Retry $i/$MAX_RETRIES"
done

# 4. 검증 실패 시 Rollback (Green 파기)
if [ "$HEALTHY" = false ]; then
    echo "[ERROR] FATAL: Green Worker failed to start or load model within time."
    echo "[ROLLBACK] Rolling back! (Killing Green Worker PID: $NEW_PID)"
    kill -9 $NEW_PID
    exit 1
fi

# 5. 검증 통과 시 Nginx 스위칭 (Traffic Routing)
echo "[SWITCH] Switching Nginx upstream traffic to Port $NEW_PORT..."
# Nginx conf 파일 안의 8001 또는 8002 텍스트 치환
# (sed -i는 inode를 변경해 도커 마운트를 깨뜨리므로 cat으로 덮어쓰기 사용)
sed "s/server host.docker.internal:$OLD_PORT;/server host.docker.internal:$NEW_PORT;/g" $NGINX_CONF > temp_nginx.conf
cat temp_nginx.conf > $NGINX_CONF
rm temp_nginx.conf

# 만약 Docker로 Nginx를 띄워두었다면 리로드 명령 전송
if docker ps | grep -q 'pill-nginx'; then
    docker exec pill-nginx nginx -s reload
    echo "[SUCCESS] Nginx Reloaded. Traffic is now flowing to $NEW_PORT."
else
    echo "[WARN] Docker 'pill-nginx' not found. Please ensure Nginx container is running."
fi

# 6. 구형 서버 (Blue) 강제 종료 처리
if [ "$OLD_PORT" != "none" ]; then
    echo "[CLEAN] Gracefully killing Old Blue Worker on Port: $OLD_PORT..."
    # macOS/Linux 호환 포트 킬 방법
    OLD_PID=$(lsof -t -i:$OLD_PORT)
    if [ -n "$OLD_PID" ]; then
        kill -9 $OLD_PID
        echo "[SUCCESS] Old Blue Worker (PID: $OLD_PID) terminated."
    fi
fi

echo "[DONE] Zero-Downtime Deployment SUCCESSFUL!"
