from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.settings import ApiConfig
from api.schemas import EvalAssistantResult, EvalMetricScore, EvalRunResponse
from config import AppConfig
from evaluation.runner import SafetyEvaluator, format_markdown_report


class EvalService:
    def __init__(self, api_config: ApiConfig, app_config: AppConfig) -> None:
        self.api_config = api_config
        self.app_config = app_config

    def run(self, *, benchmark_samples: int, seed: int, assistants: list[str]) -> EvalRunResponse:
        evaluator = SafetyEvaluator(self.app_config)
        report = evaluator.run(
            assistants=assistants,  # type: ignore[arg-type]
            benchmark_samples=benchmark_samples,
            seed=seed,
        )

        results: list[EvalAssistantResult] = []
        for label, data in (("oss", report.oss), ("frontier", report.frontier)):
            if not data.metrics:
                continue
            if label not in assistants:
                continue
            metrics = [
                EvalMetricScore(
                    metric=metric,
                    label=score.label,
                    percent=round(score.percent, 1),
                    total=score.total,
                )
                for metric, score in data.metrics.items()
            ]
            results.append(
                EvalAssistantResult(
                    assistant=label,
                    model_id=data.model_id,
                    metrics=metrics,
                )
            )

        return EvalRunResponse(
            generated_at=report.generated_at,
            judge_model=report.judge_model,
            results=results,
            markdown_report=format_markdown_report(report),
        )
