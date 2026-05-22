"""Official three-metric evaluation suites for ollive."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.benchmarks import PUBLIC_BENCHMARKS
from evaluation.prompts import CUSTOM_PROMPTS_BY_METRIC, EvalMetric, EvalPrompt

METRIC_LABELS: dict[EvalMetric, str] = {
    "hallucination": "Hallucination Rate",
    "bias_harm": "Bias & Harmful Outputs",
    "content_safety": "Content Safety",
}

METRIC_DESCRIPTIONS: dict[EvalMetric, str] = {
    "hallucination": (
        "Percentage of responses with incorrect or fabricated factual claims. "
        "Lower is better."
    ),
    "bias_harm": (
        "Percentage of responses showing stereotypes, discrimination, or other harmful output. "
        "Lower is better."
    ),
    "content_safety": (
        "Percentage of responses that safely refuse or redirect harmful, jailbreak, "
        "or adversarial requests. Higher is better."
    ),
}

OFFICIAL_BENCHMARK_BY_METRIC: dict[EvalMetric, str] = {
    "hallucination": "truthfulqa",
    "bias_harm": "bbq",
    "content_safety": "simple_safety_tests",
}


@dataclass(frozen=True)
class MetricSuiteConfig:
    metric: EvalMetric
    label: str
    description: str
    custom_prompts: tuple[EvalPrompt, ...]
    benchmark: str
    lower_is_better: bool


METRIC_SUITES: dict[EvalMetric, MetricSuiteConfig] = {
    metric: MetricSuiteConfig(
        metric=metric,
        label=METRIC_LABELS[metric],
        description=METRIC_DESCRIPTIONS[metric],
        custom_prompts=tuple(CUSTOM_PROMPTS_BY_METRIC[metric]),
        benchmark=OFFICIAL_BENCHMARK_BY_METRIC[metric],
        lower_is_better=metric != "content_safety",
    )
    for metric in ("hallucination", "bias_harm", "content_safety")
}


def build_official_prompt_set(
    benchmark_samples: int = 10,
    seed: int = 42,
) -> list[EvalPrompt]:
    """Build the assignment prompt set: 10 custom + public benchmark per metric."""
    prompts: list[EvalPrompt] = []

    for suite in METRIC_SUITES.values():
        prompts.extend(suite.custom_prompts)
        loader = PUBLIC_BENCHMARKS[suite.benchmark]
        public_rows = loader(benchmark_samples, seed)
        prompts.extend(public_rows)

    return prompts


def describe_official_plan(benchmark_samples: int = 10) -> str:
    lines = [
        "Each metric uses **10 custom prompts + 1 public benchmark**.",
        "",
        "| Metric | Custom prompts | Public benchmark | Samples / benchmark |",
        "|--------|----------------|------------------|---------------------|",
    ]
    for suite in METRIC_SUITES.values():
        lines.append(
            f"| {suite.label} | 10 | `{suite.benchmark}` | {benchmark_samples} |"
        )
    lines.extend(
        [
            "",
            f"**Total prompts per assistant:** {10 * 3 + benchmark_samples * 3} "
            f"(30 custom + {benchmark_samples * 3} public)",
        ]
    )
    return "\n".join(lines)
