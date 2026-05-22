# Hugging Face Space: FastAPI OSS service (build from repo root)
# docker build -f deploy/hf-api/Dockerfile -t ollive-api .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_HOST=0.0.0.0 \
    PORT=7860 \
    OTEL_ENABLED=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache /tmp/*

COPY api/ ./api/
COPY assistants/ ./assistants/
COPY config.py logging_config.py ./
COPY evaluation/ ./evaluation/
COPY llm/ ./llm/
COPY memory/ ./memory/
COPY tools/ ./tools/

EXPOSE 7860

CMD ["python", "api/run.py"]
