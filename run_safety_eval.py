#!/usr/bin/env python3
"""Compare OSS vs frontier assistants on the three official evaluation metrics."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import AppConfig
from evaluation.runner import SafetyEvaluator, format_markdown_report, save_report
from evaluation.suites import METRIC_SUITES, describe_official_plan
from logging_config import configure_logging

configure_logging()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official ollive evaluation: Hallucination Rate, "
            "Bias & Harmful Outputs, and Content Safety"
        )
    )
    parser.add_argument(
        "--assistant",
        action="append",
        choices=["oss", "frontier"],
        help="Run only selected assistants (default: both)",
    )
    parser.add_argument(
        "--benchmark-samples",
        type=int,
        default=10,
        help="Prompts sampled from each public benchmark (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for public benchmark sampling",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/safety_eval.json"),
        help="Path for JSON results",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/safety_eval.md"),
        help="Path for markdown summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation plan without calling models",
    )
    args = parser.parse_args()

    config = AppConfig()
    assistants = args.assistant or ["oss", "frontier"]
    evaluator = SafetyEvaluator(config)

    print("Official evaluation metrics:")
    for suite in METRIC_SUITES.values():
        print(f"  - {suite.label}: 10 custom prompts + {suite.benchmark}")
    print()
    print(describe_official_plan(args.benchmark_samples))

    prompts = evaluator.build_prompt_set(
        benchmark_samples=args.benchmark_samples,
        seed=args.seed,
    )

    if args.dry_run:
        print("\nDry run — evaluation plan")
        print(f"  OSS model: {config.oss.model_id}")
        print(f"  Frontier model: {config.frontier.model_id}")
        print(f"  Judge model: {config.judge_model_id}")
        print(f"  Assistants: {', '.join(assistants)}")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Total model runs: {len(prompts) * len(assistants)}")
        for item in prompts:
            tag = item.benchmark or "custom"
            print(f"  - [{item.metric}] [{tag}] {item.id}: {item.prompt[:90]}...")
        return

    report = evaluator.run(
        assistants=assistants,
        benchmark_samples=args.benchmark_samples,
        seed=args.seed,
    )

    save_report(report, args.output_json)
    args.output_md.write_text(format_markdown_report(report), encoding="utf-8")

    print(format_markdown_report(report))
    print(f"\nSaved JSON: {args.output_json.resolve()}")
    print(f"Saved Markdown: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
