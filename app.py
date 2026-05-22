import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistants.frontier import FrontierAssistant
from assistants.open_source import OpenSourceAssistant
from config import AppConfig
from evaluation.gradio_eval import build_evaluation_tab
from evaluation.metrics import TrackedAssistant
from logging_config import configure_logging

configure_logging()

CONFIG = AppConfig()
_assistants: dict[str, TrackedAssistant] = {}


def _get_assistant(kind: str) -> TrackedAssistant:
    if kind not in _assistants:
        if kind == "oss":
            tracked = TrackedAssistant(
                OpenSourceAssistant(CONFIG.oss, CONFIG.oss_system_prompt),
                name="Open Source Assistant",
            )
        elif kind == "frontier":
            tracked = TrackedAssistant(
                FrontierAssistant(CONFIG.frontier, CONFIG.system_prompt),
                name="Frontier Model Assistant",
            )
        else:
            raise ValueError(f"Unknown assistant kind: {kind}")
        _assistants[kind] = tracked
    return _assistants[kind]


def _format_chat_history(history: list[dict] | None) -> list[dict]:
    return history or []


def _compare() -> str:
    oss = _assistants.get("oss")
    frontier = _assistants.get("frontier")
    oss_summary = oss.metrics.summary() if oss else "**Open Source Assistant**\n- No session yet"
    frontier_summary = (
        frontier.metrics.summary() if frontier else "**Frontier Model Assistant**\n- No session yet"
    )
    return f"{oss_summary}\n\n---\n\n{frontier_summary}"


def _chat(
    user_message: str,
    history: list[dict] | None,
    assistant: TrackedAssistant,
) -> tuple[list[dict], str, str, list[dict]]:
    history = _format_chat_history(history)
    if not user_message.strip():
        return history, "", assistant.metrics.summary(), history

    response = assistant.chat(user_message)
    reply = response.text.strip() or "(No response generated)"
    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    footer = (
        f"_{response.latency_ms:.0f} ms · {response.model}_"
        if response.error is None
        else f"_Error · {response.model}_"
    )
    return history, "", f"{assistant.metrics.summary()}\n\n{footer}", history


def _chat_oss(msg: str, hist: list[dict] | None) -> tuple[list[dict], str, str, list[dict]]:
    return _chat(msg, hist, _get_assistant("oss"))


def _chat_frontier(msg: str, hist: list[dict] | None) -> tuple[list[dict], str, str, list[dict]]:
    return _chat(msg, hist, _get_assistant("frontier"))


def _reset(assistant: TrackedAssistant) -> tuple[list[dict], str, str, list[dict]]:
    assistant.reset()
    return [], "", assistant.metrics.summary(), []


def _reset_oss() -> tuple[list[dict], str, str, list[dict]]:
    if "oss" in _assistants:
        return _reset(_assistants["oss"])
    return [], "", "**Open Source Assistant**\n- Model: `n/a`\n- Turns: 0", []


def _reset_frontier() -> tuple[list[dict], str, str, list[dict]]:
    if "frontier" in _assistants:
        return _reset(_assistants["frontier"])
    return [], "", "**Frontier Model Assistant**\n- Model: `n/a`\n- Turns: 0", []


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="ollive — AI Personal Assistants") as demo:
        gr.Markdown(
            "# ollive — AI Personal Assistants\n"
            "Two LangChain assistants with the same capabilities: multi-turn chat, "
            "short-term memory, and tool-calling support (tools can be added in "
            "`tools/registry.py`).\n\n"
            f"**OSS model:** `{CONFIG.oss.model_id}` · "
            f"**Frontier model:** `{CONFIG.frontier.model_id}` "
            f"({CONFIG.frontier.provider})"
        )

        with gr.Tabs():
            with gr.Tab("Open Source Assistant"):
                gr.Markdown(
                    "Powered by LangChain + local Hugging Face model "
                    "(`OSS_BACKEND=local`, default). Optional `HF_TOKEN` for gated models."
                )
                oss_history = gr.State([])
                oss_chat = gr.Chatbot(height=420)
                oss_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask anything…",
                    lines=2,
                )
                with gr.Row():
                    oss_send = gr.Button("Send", variant="primary")
                    oss_clear = gr.Button("Clear memory")
                oss_metrics = gr.Markdown("**Open Source Assistant**\n- Ready")

                oss_send.click(
                    fn=_chat_oss,
                    inputs=[oss_input, oss_history],
                    outputs=[oss_chat, oss_input, oss_metrics, oss_history],
                )
                oss_input.submit(
                    fn=_chat_oss,
                    inputs=[oss_input, oss_history],
                    outputs=[oss_chat, oss_input, oss_metrics, oss_history],
                )
                oss_clear.click(
                    fn=_reset_oss,
                    outputs=[oss_chat, oss_input, oss_metrics, oss_history],
                )

            with gr.Tab("Frontier Model Assistant"):
                gr.Markdown(
                    "Powered by LangChain + hosted API. Requires `OPENAI_API_KEY` "
                    "(or `ANTHROPIC_API_KEY` with `FRONTIER_PROVIDER=anthropic`)."
                )
                frontier_history = gr.State([])
                frontier_chat = gr.Chatbot(height=420)
                frontier_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask anything…",
                    lines=2,
                )
                with gr.Row():
                    frontier_send = gr.Button("Send", variant="primary")
                    frontier_clear = gr.Button("Clear memory")
                frontier_metrics = gr.Markdown("**Frontier Model Assistant**\n- Ready")

                frontier_send.click(
                    fn=_chat_frontier,
                    inputs=[frontier_input, frontier_history],
                    outputs=[frontier_chat, frontier_input, frontier_metrics, frontier_history],
                )
                frontier_input.submit(
                    fn=_chat_frontier,
                    inputs=[frontier_input, frontier_history],
                    outputs=[frontier_chat, frontier_input, frontier_metrics, frontier_history],
                )
                frontier_clear.click(
                    fn=_reset_frontier,
                    outputs=[frontier_chat, frontier_input, frontier_metrics, frontier_history],
                )

            with gr.Tab("Compare"):
                gr.Markdown("Side-by-side session metrics from your current chats.")
                compare_btn = gr.Button("Refresh comparison", variant="secondary")
                compare_out = gr.Markdown("")

                compare_btn.click(fn=_compare, outputs=[compare_out])

            build_evaluation_tab(CONFIG)

    return demo


def main() -> None:
    import os

    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )


if __name__ == "__main__":
    main()
