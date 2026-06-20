#!/usr/bin/env bash
set -euo pipefail

# [Design Intent]
# 수동 커맨드 반복을 줄이기 위해 후보 등록 -> staging 헬스체크 -> (선택) promote -> (선택) deploy를
# 하나의 스크립트로 묶는다.

MODEL_NAME="pill_detection_2026"
RUN_ID=""
STAGING_PORT="18002"
MAX_RETRIES="30"
AUTO_PROMOTE="false"
AUTO_DEPLOY="false"
BUILD_IMAGE="true"
IMAGE_NAME="pill-api-image"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/staging_validate_and_release.sh --run-id <RUN_ID> [options]

Required:
  --run-id <id>                MLflow run_id to register as new candidate version

Options:
  --model-name <name>          Registered model name (default: pill_detection_2026)
  --staging-port <port>        Local validation port (default: 18002)
  --max-retries <n>            Health check retries, 2s interval (default: 30)
  --auto-promote               Promote @staging -> @production after validation
  --auto-deploy                Run deploy.sh with MODEL_URI=@production after promote
  --skip-build                 Do not build docker image before validation
  -h, --help                   Show help

Examples:
  bash scripts/staging_validate_and_release.sh --run-id abc123
  bash scripts/staging_validate_and_release.sh --run-id abc123 --auto-promote
  bash scripts/staging_validate_and_release.sh --run-id abc123 --auto-promote --auto-deploy
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="${2:-}"
      shift 2
      ;;
    --staging-port)
      STAGING_PORT="${2:-}"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="${2:-}"
      shift 2
      ;;
    --auto-promote)
      AUTO_PROMOTE="true"
      shift
      ;;
    --auto-deploy)
      AUTO_DEPLOY="true"
      shift
      ;;
    --skip-build)
      BUILD_IMAGE="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "[ERROR] --run-id is required"
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/5] Register candidate version and set @staging"
python scripts/registry_ops.py register --model-name "$MODEL_NAME" --run-id "$RUN_ID"

if [[ "$BUILD_IMAGE" == "true" ]]; then
  echo "[2/5] Build serving image: $IMAGE_NAME"
  docker build -t "$IMAGE_NAME" . >/dev/null
else
  echo "[2/5] Skip image build"
fi

CONTAINER_NAME="staging-check-${STAGING_PORT}"
MODEL_URI="models:/${MODEL_NAME}@staging"
HOST_PROJECT_DIR="$(pwd)"
TRACKING_URI="file:$HOST_PROJECT_DIR/mlruns"

echo "[3/5] Resolve staging alias URI -> local checkpoint path"
RESOLVED_MODEL_URI="$(python scripts/resolve_registry_model_path.py --model-uri "$MODEL_URI" --tracking-uri "$TRACKING_URI")"
echo "       resolved: $RESOLVED_MODEL_URI"

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[3/5] Start temporary staging container: $CONTAINER_NAME (port ${STAGING_PORT})"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${STAGING_PORT}:8000" \
  -v "$HOST_PROJECT_DIR:$HOST_PROJECT_DIR:rw" \
  -e MLFLOW_TRACKING_URI="$TRACKING_URI" \
  -e MODEL_URI="$RESOLVED_MODEL_URI" \
  "$IMAGE_NAME" >/dev/null

echo "[4/5] Health check: http://127.0.0.1:${STAGING_PORT}/health"
HEALTHY="false"
for i in $(seq 1 "$MAX_RETRIES"); do
  sleep 2
  STATUS_CODE="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${STAGING_PORT}/health" || true)"
  if [[ "$STATUS_CODE" == "200" ]]; then
    HEALTHY="true"
    echo "[OK] staging health check passed"
    break
  fi
  echo "  ...waiting status=${STATUS_CODE:-NA} retry ${i}/${MAX_RETRIES}"
done

if [[ "$HEALTHY" != "true" ]]; then
  echo "[ERROR] staging health check failed"
  echo "[DEBUG] container logs:"
  docker logs --tail 120 "$CONTAINER_NAME" || true
  exit 1
fi

if [[ "$AUTO_PROMOTE" == "true" ]]; then
  echo "[5/5] Promote staging -> production"
  python scripts/registry_ops.py promote --model-name "$MODEL_NAME"

  if [[ "$AUTO_DEPLOY" == "true" ]]; then
    echo "[5/5+] Deploy production alias via deploy.sh"
    export MODEL_URI="models:/${MODEL_NAME}@production"
    bash scripts/deploy.sh
  fi
else
  echo "[5/5] Skip promote/deploy (validation only)"
fi

echo "[DONE] Pipeline complete"
