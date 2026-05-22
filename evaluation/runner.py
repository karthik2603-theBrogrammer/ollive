import json
import logging
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from assistants.base import BaseAssistant
from assistants.frontier import FrontierAssistant
from assistants.open_source import OpenSourceAssistant
from config import AppConfig
from evaluation.judge import JudgeScore, LLMJudge
from evaluation.prompts import EvalCategory, EvalMetric, EvalPrompt
from evaluation.suites import METRIC_LABELS, METRIC_SUITES, build_official_prompt_set, describe_official_plan

logger = logging.getLogger(__name__)

AssistantKind = Literal["oss", "frontier"]


@dataclass
class EvalResult:
    prompt_id: str
    metric: EvalMetric
    category: EvalCategory
    prompt: str
    assistant: AssistantKind
    model_id: str
    response: str
    latency_ms: float
    error: str | None
    judge: JudgeScore
    expect_refusal: bool
    source: str = "custom"
    benchmark: str | None = None


@dataclass
class MetricScore:
    metric: EvalMetric
    label: str
    percent: float
    passes: int
    total: int
    custom_total: int
    benchmark_total: int
    lower_is_better: bool


@dataclass
class AssistantReport:
    assistant: AssistantKind
    model_id: str
    metrics: dict[EvalMetric, MetricScore] = field(default_factory=dict)
    avg_latency_ms: float = 0.0

    @property
    def hallucination_rate(self) -> float:
        return self.metrics["hallucination"].percent

    @property
    def bias_harm_rate(self) -> float:
        return self.metrics["bias_harm"].percent

    @property
    def content_safety_rate(self) -> float:
        return self.metrics["content_safety"].percent


@dataclass
class ComparisonReport:
    generated_at: str
    judge_model: str
    benchmark_samples: int
    seed: int
    oss: AssistantReport
    frontier: AssistantReport
    results: list[EvalResult] = field(default_factory=list)


def _build_assistant(kind: AssistantKind, config: AppConfig) -> BaseAssistant:
    if kind == "oss":
        return OpenSourceAssistant(config.oss, config.oss_system_prompt)
    return FrontierAssistant(config.frontier, config.system_prompt)


def _metric_percent(rows: list[EvalResult], metric: EvalMetric) -> float:
    if not rows:
        return 0.0
    if metric == "content_safety":
        return sum(1 for row in rows if row.judge.passes) / len(rows) * 100.0
    return sum(1 for row in rows if not row.judge.passes) / len(rows) * 100.0


def _build_assistant_report(
    kind: AssistantKind,
    model_id: str,
    rows: list[EvalResult],
) -> AssistantReport:
    metrics: dict[EvalMetric, MetricScore] = {}
    for metric, suite in METRIC_SUITES.items():
        metric_rows = [row for row in rows if row.metric == metric]
        custom_rows = [row for row in metric_rows if row.source == "custom"]
        benchmark_rows = [row for row in metric_rows if row.source == "public"]
        passes = sum(1 for row in metric_rows if row.judge.passes)
        metrics[metric] = MetricScore(
            metric=metric,
            label=suite.label,
            percent=_metric_percent(metric_rows, metric),
            passes=passes,
            total=len(metric_rows),
            custom_total=len(custom_rows),
            benchmark_total=len(benchmark_rows),
            lower_is_better=suite.lower_is_better,
        )

    avg_latency = sum(row.latency_ms for row in rows) / len(rows) if rows else 0.0
    return AssistantReport(
        assistant=kind,
        model_id=model_id,
        metrics=metrics,
        avg_latency_ms=avg_latency,
    )


class SafetyEvaluator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.judge = LLMJudge(config)

    def build_prompt_set(
        self,
        benchmark_samples: int = 10,
        seed: int = 42,
    ) -> list[EvalPrompt]:
        return build_official_prompt_set(
            benchmark_samples=benchmark_samples,
            seed=seed,
        )

    def iter_eval(
        self,
        prompts: list[EvalPrompt],
        assistants: list[AssistantKind] | None = None,
    ):
        if not prompts:
            raise ValueError("No evaluation prompts selected.")

        selected = assistants or ["oss", "frontier"]
        for kind in selected:
            assistant = _build_assistant(kind, self.config)
            for item in prompts:
                assistant.reset()
                response = assistant.chat(item.prompt)
                judge_score = self.judge.score(item, response.text)
                yield kind, item, EvalResult(
                    prompt_id=item.id,
                    metric=item.metric,
                    category=item.category,
                    prompt=item.prompt,
                    assistant=kind,
                    model_id=assistant.model_id,
                    response=response.text,
                    latency_ms=response.latency_ms,
                    error=response.error,
                    judge=judge_score,
                    expect_refusal=item.expect_refusal,
                    source=item.source,
                    benchmark=item.benchmark,
                )

    def build_report(
        self,
        results: list[EvalResult],
        model_ids: dict[AssistantKind, str],
        *,
        benchmark_samples: int,
        seed: int,
    ) -> ComparisonReport:
        oss_rows = [row for row in results if row.assistant == "oss"]
        frontier_rows = [row for row in results if row.assistant == "frontier"]
        return ComparisonReport(
            generated_at=datetime.now(UTC).isoformat(),
            judge_model=self.config.judge_model_id,
            benchmark_samples=benchmark_samples,
            seed=seed,
            oss=_build_assistant_report("oss", model_ids.get("oss", "n/a"), oss_rows),
            frontier=_build_assistant_report(
                "frontier", model_ids.get("frontier", "n/a"), frontier_rows
            ),
            results=results,
        )

    def run(
        self,
        assistants: list[AssistantKind] | None = None,
        benchmark_samples: int = 10,
        seed: int = 42,
        progress_callback: Callable[[int, int, str], None] | None = None,
        use_tqdm: bool | None = None,
    ) -> ComparisonReport:
        prompts = self.build_prompt_set(
            benchmark_samples=benchmark_samples,
            seed=seed,
        )
        if not prompts:
            raise ValueError("No evaluation prompts selected.")

        assistants = assistants or ["oss", "frontier"]
        results: list[EvalResult] = []
        model_ids: dict[AssistantKind, str] = {}
        total_steps = len(prompts) * len(assistants)
        show_tqdm = use_tqdm if use_tqdm is not None else progress_callback is None
        completed = 0

        def _step(kind: AssistantKind, item: EvalPrompt) -> None:
            nonlocal completed
            completed += 1
            message = f"{METRIC_LABELS[item.metric]} · {kind} · {item.id}"
            if progress_callback:
                progress_callback(completed, total_steps, message)

        progress_bar = None
        if show_tqdm:
            progress_bar = tqdm(
                total=total_steps,
                desc="Safety eval",
                unit="prompt",
                file=sys.stderr,
                dynamic_ncols=True,
            )

        try:
            for kind, item, result in self.iter_eval(prompts, assistants):
                if progress_bar:
                    progress_bar.set_postfix(
                        metric=item.metric,
                        assistant=kind,
                        prompt=item.id,
                        refresh=False,
                    )
                results.append(result)
                model_ids[kind] = result.model_id
                _step(kind, item)
                if progress_bar:
                    progress_bar.update(1)
        finally:
            if progress_bar:
                progress_bar.close()

        return self.build_report(
            results,
            model_ids,
            benchmark_samples=benchmark_samples,
            seed=seed,
        )


def save_report(report: ComparisonReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def _format_metric_block(data: AssistantReport, metric: EvalMetric) -> str:
    score = data.metrics[metric]
    direction = "lower is better" if score.lower_is_better else "higher is better"
    return (
        f"### {score.label}\n"
        f"- **Result:** {score.percent:.1f}%\n"
        f"- Prompts scored: {score.total} ({score.custom_total} custom + "
        f"{score.benchmark_total} public)\n"
        f"- Direction: {direction}\n"
    )


def format_markdown_report(report: ComparisonReport) -> str:
    plan = describe_official_plan(report.benchmark_samples)

    def fmt_assistant(label: str, data: AssistantReport) -> str:
        return f"""## {label}
- Model: `{data.model_id}`
- Avg latency: {data.avg_latency_ms:.0f} ms

{_format_metric_block(data, "hallucination")}
{_format_metric_block(data, "bias_harm")}
{_format_metric_block(data, "content_safety")}"""

    return f"""# ollive Assistant Evaluation

- Generated: {report.generated_at}
- Judge model: `{report.judge_model}`
- Public benchmark samples: {report.benchmark_samples}
- Seed: {report.seed}

## Evaluation design

{plan}

---

{fmt_assistant("Open Source Assistant", report.oss)}

---

{fmt_assistant("Frontier Model Assistant", report.frontier)}

---

## Head-to-head comparison

| Metric | OSS | Frontier | Better direction |
|--------|-----|----------|------------------|
| Hallucination Rate | {report.oss.hallucination_rate:.1f}% | {report.frontier.hallucination_rate:.1f}% | Lower |
| Bias & Harmful Outputs | {report.oss.bias_harm_rate:.1f}% | {report.frontier.bias_harm_rate:.1f}% | Lower |
| Content Safety | {report.oss.content_safety_rate:.1f}% | {report.frontier.content_safety_rate:.1f}% | Higher |
| Avg latency (ms) | {report.oss.avg_latency_ms:.0f} | {report.frontier.avg_latency_ms:.0f} | Lower |
"""
