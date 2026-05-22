# Hugging Face Space: Gradio UI (build from repo root)
# docker build -f deploy/hf-gradio/Dockerfile -t ollive-gradio .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

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

COPY app.py config.py logging_config.py ./
COPY assistants/ ./assistants/
COPY evaluation/ ./evaluation/
COPY llm/ ./llm/
COPY memory/ ./memory/
COPY tools/ ./tools/

EXPOSE 7860

CMD ["python", "app.py"]
