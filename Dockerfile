# [Design Intent]: 프로덕션급 추론 서비스를 위한 경량 도커 이미지
FROM python:3.12-slim

# 보안 및 성능을 위한 파이썬 환경변수 최적화
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# 시스템 단위 의존성(OpenCV에 필수적인 라이브러리들)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 의존성 복사 및 설치 (도커 레이어 캐싱 활용)
# Stage A/B 분리에 따라 requirements-runtime.txt를 사용
COPY requirements-runtime.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스코드 복사
COPY . .

# FastAPI 서버 실행 (Gunicorn + UvicornWorker 조합으로 비동기 성능 극대화)
# - bind: 컨테이너 내부는 항상 8000번 포트로 통일 (deploy.sh에서 외부 포트 매핑)
# - workers: 단일 워커 (컨테이너 자체를 여러 개 띄우는 것이 정석이므로 1개로 제한)
CMD ["gunicorn", "src.api.main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "--graceful-timeout", "30"]
