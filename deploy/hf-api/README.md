---
title: Ollive OSS API
emoji: 🫒
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ollive OSS API

FastAPI service: chat, guardrails, tools, OpenTelemetry traces UI, inference metrics.

| Endpoint | Description |
|----------|-------------|
| `/docs` | Swagger UI |
| `/v1/chat` | Chat API |
| `/v1/traces/ui` | Observability dashboard |
| `/v1/metrics/inference` | TTFT, TBT, latency |

## Space setup (monorepo)

1. Create a **Docker** Space on Hugging Face.
2. Connect this GitHub repo.
3. Set **Space directory** (subpath) to: `deploy/hf-api`
4. Add **Secrets**: `HF_TOKEN` (gated models), optional `API_SERVICE_KEY`.

## Local build

```bash
docker build -f deploy/hf-api/Dockerfile -t ollive-api .
docker run --rm -p 7860:7860 --env-file .env ollive-api
```

Open http://localhost:7860/v1/traces/ui
