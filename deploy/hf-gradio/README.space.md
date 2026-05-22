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

Multi-tab Gradio app: OSS assistant (Qwen), frontier assistant, compare, and safety eval.

## Secrets (Space settings)

| Secret | Required |
|--------|----------|
| `HF_TOKEN` | Yes (gated models) |
| `OPENAI_API_KEY` | Optional (frontier + eval judge) |
| `ANTHROPIC_API_KEY` | Optional if `FRONTIER_PROVIDER=anthropic` |
