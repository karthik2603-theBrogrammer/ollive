#!/usr/bin/env python3
"""Run a scripted evaluation comparing OSS vs frontier assistants."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from assistants.frontier import FrontierAssistant
from assistants.open_source import OpenSourceAssistant
from config import AppConfig
from evaluation.metrics import TrackedAssistant
from logging_config import configure_logging

configure_logging(fmt="%(levelname)s | %(message)s")

DEFAULT_PROMPTS = [
    "Hi, my name is Alex. Remember that for this conversation.",
    "What's my name?",
    "Suggest three healthy lunch ideas under 15 minutes.",
    "Which of those would work best if I'm avoiding dairy?",
]


def run_eval(prompts: list[str], dry_run: bool = False) -> None:
    config = AppConfig()

    if dry_run:
        print("Dry run — validating config only")
        print(f"  OSS model: {config.oss.model_id}")
        print(f"  Frontier model: {config.frontier.model_id} ({config.frontier.provider})")
        for prompt in prompts:
            print(f"\nUser: {prompt}")
            print("Assistant: [dry-run — skipped API call]")
        return

    assistants = [
        TrackedAssistant(
            OpenSourceAssistant(config.oss, config.oss_system_prompt),
            name="Open Source",
        ),
        TrackedAssistant(
            FrontierAssistant(config.frontier, config.system_prompt),
            name="Frontier",
        ),
    ]

    for tracked in assistants:
        print(f"\n{'=' * 60}\nEvaluating: {tracked.metrics.name}\n{'=' * 60}")
        for prompt in prompts:
            print(f"\nUser: {prompt}")
            response = tracked.chat(prompt)
            print(f"Assistant ({response.latency_ms:.0f} ms): {response.text[:500]}")
            if response.error:
                print(f"  ERROR: {response.error}")
        print(f"\n{tracked.metrics.summary()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ollive assistants")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling APIs",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Custom prompt (repeatable)",
    )
    args = parser.parse_args()
    prompts = args.prompts or DEFAULT_PROMPTS
    run_eval(prompts, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
