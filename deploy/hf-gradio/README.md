---
title: Ollive Gradio
emoji: 🫒
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Ollive — Gradio UI

Multi-tab Gradio app: OSS assistant (local Qwen), frontier assistant, compare, and safety eval.

## Space setup (monorepo)

1. Create a **Docker** Space on Hugging Face.
2. Connect this GitHub repo.
3. Set **Space directory** (subpath) to: `deploy/hf-gradio`
4. Add **Secrets**: `HF_TOKEN`, `OPENAI_API_KEY` (optional, for frontier/eval).

## Local build

```bash
docker build -f deploy/hf-gradio/Dockerfile -t ollive-gradio .
docker run --rm -p 7860:7860 --env-file .env ollive-gradio
```

Open http://localhost:7860
