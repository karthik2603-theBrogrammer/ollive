"""Gradio UI handlers for the official three-metric evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

from config import AppConfig
from evaluation.runner import AssistantKind, SafetyEvaluator, format_markdown_report, save_report
from evaluation.suites import METRIC_SUITES, describe_official_plan

ASSISTANT_CHOICES = ["oss", "frontier"]


def _normalize_assistants(assistants: list[str]) -> list[AssistantKind]:
    if not assistants:
        return ASSISTANT_CHOICES.copy()
    return [name for name in assistants if name in ASSISTANT_CHOICES]  # type: ignore[misc]


def preview_evaluation_plan(
    assistants: list[str],
    benchmark_samples: int,
    seed: int,
    config: AppConfig,
) -> str:
    selected_assistants = _normalize_assistants(assistants)
    evaluator = SafetyEvaluator(config)
    prompts = evaluator.build_prompt_set(
        benchmark_samples=int(benchmark_samples),
        seed=int(seed),
    )

    lines = [
        describe_official_plan(int(benchmark_samples)),
        "",
        f"**Assistants:** {', '.join(selected_assistants)}",
        f"**Total prompts:** {len(prompts)}",
        f"**Total model runs:** {len(prompts) * len(selected_assistants)}",
        f"**Judge model:** `{config.judge_model_id}`",
        "",
        "### Prompt preview",
    ]

    for item in prompts[:30]:
        tag = item.benchmark or "custom"
        lines.append(
            f"- **{METRIC_SUITES[item.metric].label}** `[{tag}]` "
            f"`{item.id}`: {item.prompt[:90]}..."
        )
    if len(prompts) > 30:
        lines.append(f"- _…and {len(prompts) - 30} more_")

    return "\n".join(lines)


def run_evaluation_ui(
    assistants: list[str],
    benchmark_samples: int,
    seed: int,
    save_results: bool,
    config: AppConfig,
    progress: gr.Progress | None = None,
):
    selected_assistants = _normalize_assistants(assistants)
    if not selected_assistants:
        raise gr.Error("Select at least one assistant to evaluate.")

    evaluator = SafetyEvaluator(config)
    prompts = evaluator.build_prompt_set(
        benchmark_samples=int(benchmark_samples),
        seed=int(seed),
    )
    if not prompts:
        raise gr.Error("No evaluation prompts were loaded.")

    total_steps = len(prompts) * len(selected_assistants)
    tracker = progress or gr.Progress()
    results = []
    model_ids: dict[AssistantKind, str] = {}

    yield "", "_Starting evaluation…_", None, 0.0

    completed = 0
    for kind, item, result in evaluator.iter_eval(prompts, selected_assistants):
        completed += 1
        message = f"{METRIC_SUITES[item.metric].label} · {kind} · {item.id}"
        model_ids[kind] = result.model_id
        results.append(result)
        pct = completed / total_steps
        tracker(pct, desc=message, total=total_steps)
        yield "", f"**Progress:** {completed}/{total_steps} — `{message}`", None, pct

    report = evaluator.build_report(
        results,
        model_ids,
        benchmark_samples=int(benchmark_samples),
        seed=int(seed),
    )
    markdown = format_markdown_report(report)
    saved_path: str | None = None
    if save_results:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        json_path = results_dir / f"eval_ui_{stamp}.json"
        md_path = results_dir / f"eval_ui_{stamp}.md"
        save_report(report, json_path)
        md_path.write_text(markdown, encoding="utf-8")
        saved_path = str(json_path.resolve())

    status = (
        f"Completed **{len(report.results)}** scored responses across "
        f"**{len(selected_assistants)}** assistants."
    )
    if saved_path:
        status += f" Saved to `{saved_path}`."

    yield markdown, status, saved_path, 1.0


def build_evaluation_tab(config: AppConfig) -> None:
    metric_lines = "\n".join(
        f"- **{suite.label}:** 10 custom prompts + `{suite.benchmark}`"
        for suite in METRIC_SUITES.values()
    )

    with gr.Tab("Evaluation"):
        gr.Markdown(
            "### Official three-metric evaluation\n"
            "Compare OSS vs frontier on exactly three percentages:\n"
            f"{metric_lines}\n\n"
            f"LLM-as-judge: `{config.judge_model_id}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                assistants = gr.CheckboxGroup(
                    label="Assistants",
                    choices=ASSISTANT_CHOICES,
                    value=ASSISTANT_CHOICES,
                )
                benchmark_samples = gr.Slider(
                    label="Public benchmark samples per metric",
                    minimum=1,
                    maximum=20,
                    step=1,
                    value=10,
                )
                seed = gr.Number(label="Random seed", value=42, precision=0)
                save_results = gr.Checkbox(
                    label="Save JSON + Markdown to results/",
                    value=True,
                )

                with gr.Row():
                    preview_btn = gr.Button("Preview plan", variant="secondary")
                    run_btn = gr.Button("Run evaluation", variant="primary")

            with gr.Column(scale=2):
                status = gr.Markdown("_Configure settings and run evaluation._")
                progress_pct = gr.Slider(
                    label="Evaluation progress",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0,
                    interactive=False,
                )
                report = gr.Markdown("")
                download = gr.File(label="Download JSON report", interactive=False)

        shared_inputs = [assistants, benchmark_samples, seed]

        preview_btn.click(
            fn=lambda *args: (
                preview_evaluation_plan(*args, config=config),
                "_Preview ready — no models called._",
                None,
                0.0,
            ),
            inputs=shared_inputs,
            outputs=[report, status, download, progress_pct],
        )

        def _run_eval(
            assistants_value,
            benchmark_samples_value,
            seed_value,
            save_results_value,
            progress=gr.Progress(),
        ):
            yield from run_evaluation_ui(
                assistants=assistants_value,
                benchmark_samples=benchmark_samples_value,
                seed=seed_value,
                save_results=save_results_value,
                config=config,
                progress=progress,
            )

        run_btn.click(
            fn=_run_eval,
            inputs=[*shared_inputs, save_results],
            outputs=[report, status, download, progress_pct],
        )
