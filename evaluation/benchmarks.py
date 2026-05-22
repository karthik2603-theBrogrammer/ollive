"""Load public safety/factuality benchmarks from Hugging Face datasets."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable

from evaluation.prompts import EvalMetric, EvalPrompt

logger = logging.getLogger(__name__)

BenchmarkLoader = Callable[[int, int], list[EvalPrompt]]


def _sample_rows(rows: list[EvalPrompt], limit: int, seed: int) -> list[EvalPrompt]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def _get_row_field(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _truthfulqa_reference(row: dict) -> str | None:
    best = _get_row_field(row, "best_answer", "Best Answer")
    correct = _get_row_field(row, "correct_answers", "Correct Answers")
    if isinstance(correct, str):
        parts = [part.strip() for part in correct.split(";") if part.strip()]
        correct = parts
    elif not correct:
        correct = []
    if best:
        return str(best).strip()
    if correct:
        return str(correct[0]).strip()
    return None


def load_truthfulqa(limit: int = 5, seed: int = 42) -> list[EvalPrompt]:
    """TruthfulQA (Lin et al.) — public factuality / hallucination benchmark."""
    from datasets import load_dataset

    dataset_candidates = [
        ("domenicrosati/TruthfulQA", None, "train"),
        ("truthfulqa/truthful_qa", "generation", "validation"),
        ("leibni/truthful_qa", "generation", "validation"),
    ]

    for dataset_name, config, split in dataset_candidates:
        try:
            if config:
                ds = load_dataset(dataset_name, config, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)

            prompts: list[EvalPrompt] = []
            for idx, row in enumerate(ds):
                question = _get_row_field(row, "question", "Question")
                if not question:
                    continue
                prompts.append(
                    EvalPrompt(
                        id=f"truthfulqa_{idx}",
                        metric="hallucination",
                        category="factual",
                        prompt=str(question).strip(),
                        reference=_truthfulqa_reference(row),
                        expect_refusal=False,
                        notes=f"TruthfulQA from {dataset_name}",
                        source="public",
                        benchmark="truthfulqa",
                    )
                )

            sampled = _sample_rows(prompts, limit, seed)
            if sampled:
                logger.info("Loaded TruthfulQA from %s", dataset_name)
                return sampled
        except Exception as exc:
            logger.warning("Could not load TruthfulQA from %s: %s", dataset_name, exc)

    logger.warning("All TruthfulQA sources failed; returning empty set")
    return []


def load_real_toxicity_prompts(limit: int = 5, seed: int = 42) -> list[EvalPrompt]:
    """RealToxicityPrompts (Gehman et al.) — toxicity / harmful generation benchmark."""
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    prompts: list[EvalPrompt] = []

    for idx, row in enumerate(ds):
        text = row.get("prompt", {}).get("text") or row.get("text") or ""
        text = str(text).strip()
        if not text:
            continue
        prompts.append(
            EvalPrompt(
                id=f"rtp_{idx}",
                metric="bias_harm",
                category="bias",
                prompt=text,
                expect_refusal=True,
                notes="RealToxicityPrompts",
                source="public",
                benchmark="real_toxicity_prompts",
            )
        )

    return _sample_rows(prompts, limit, seed)


def _extract_prompt_text(row: dict) -> str:
    for key in (
        "prompt",
        "goal",
        "question",
        "instruction",
        "text",
        "adversarial",
        "vanilla",
        "jailbreak_query",
        "behavior",
    ):
        value = row.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _rows_to_adversarial_prompts(
    rows,
    *,
    benchmark_name: str,
    source_note: str,
    id_prefix: str,
) -> list[EvalPrompt]:
    prompts: list[EvalPrompt] = []
    for idx, row in enumerate(rows):
        text = _extract_prompt_text(row)
        if not text:
            continue
        prompts.append(
            EvalPrompt(
                id=f"{id_prefix}_{idx}",
                metric="content_safety",
                category="adversarial",
                prompt=text,
                expect_refusal=True,
                notes=source_note,
                source="public",
                benchmark=benchmark_name,
            )
        )
    return prompts


def load_simple_safety_tests(limit: int = 5, seed: int = 42) -> list[EvalPrompt]:
    """Safety/jailbreak benchmarks — SimpleSafetyTests + open fallbacks."""
    from datasets import load_dataset

    def load_sst_instruct() -> list[EvalPrompt]:
        ds = load_dataset("walledai/SimpleSafetyTests", split="instruct")
        return _rows_to_adversarial_prompts(
            ds,
            benchmark_name="simple_safety_tests",
            source_note="walledai/SimpleSafetyTests (instruct)",
            id_prefix="sst",
        )

    def load_jailbreak_wild() -> list[EvalPrompt]:
        ds = load_dataset(
            "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "jailbreak_2023_05_07",
            split="train",
        )
        return _rows_to_adversarial_prompts(
            ds,
            benchmark_name="simple_safety_tests",
            source_note="TrustAIRLab/in-the-wild-jailbreak-prompts",
            id_prefix="jailbreak",
        )

    def load_forbidden_questions() -> list[EvalPrompt]:
        ds = load_dataset("TrustAIRLab/forbidden_question_set", split="train")
        return _rows_to_adversarial_prompts(
            ds,
            benchmark_name="simple_safety_tests",
            source_note="TrustAIRLab/forbidden_question_set",
            id_prefix="forbidden",
        )

    for name, loader in [
        ("SimpleSafetyTests", load_sst_instruct),
        ("in-the-wild-jailbreak-prompts", load_jailbreak_wild),
        ("forbidden_question_set", load_forbidden_questions),
    ]:
        try:
            prompts = loader()
            sampled = _sample_rows(prompts, limit, seed)
            if sampled:
                logger.info("Loaded safety benchmark from %s", name)
                return sampled
        except Exception as exc:
            logger.warning("Could not load safety benchmark %s: %s", name, exc)

    logger.warning("All safety benchmark sources failed; returning empty set")
    return []


def _load_bbq_subset(dataset_name: str, config: str | None, limit: int, seed: int) -> list[EvalPrompt]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(dataset_name, config, split="test")
    else:
        ds = load_dataset(dataset_name, split="test")

    prompts: list[EvalPrompt] = []
    for idx, row in enumerate(ds):
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        if not question:
            continue
        prompt = f"{context}\n\n{question}".strip() if context else question
        prompts.append(
            EvalPrompt(
                id=f"bbq_{idx}",
                metric="bias_harm",
                category="bias",
                prompt=prompt,
                expect_refusal=False,
                notes=f"BBQ subset from {dataset_name}",
                source="public",
                benchmark="bbq",
            )
        )
    return _sample_rows(prompts, limit, seed)


def load_bbq(limit: int = 5, seed: int = 42) -> list[EvalPrompt]:
    """BBQ (Bias Benchmark for QA, Parrish et al.) — social bias benchmark."""
    candidates = [
        ("oskarvanderwal/bbq", "Age"),
        ("Heegyu/bbq", None),
    ]
    last_error: Exception | None = None

    for dataset_name, config in candidates:
        try:
            return _load_bbq_subset(dataset_name, config, limit, seed)
        except Exception as exc:
            last_error = exc
            logger.warning("Could not load BBQ subset %s: %s", dataset_name, exc)

    raise RuntimeError("Failed to load BBQ benchmark") from last_error


PUBLIC_BENCHMARKS: dict[str, BenchmarkLoader] = {
    "truthfulqa": load_truthfulqa,
    "real_toxicity_prompts": load_real_toxicity_prompts,
    "simple_safety_tests": load_simple_safety_tests,
    "bbq": load_bbq,
}


def load_public_benchmarks(
    names: list[str] | None = None,
    samples_per_benchmark: int = 5,
    seed: int = 42,
) -> list[EvalPrompt]:
    if names and "all" in names:
        names = list(PUBLIC_BENCHMARKS.keys())
    selected = names or list(PUBLIC_BENCHMARKS.keys())
    prompts: list[EvalPrompt] = []
    failures: list[str] = []

    for name in selected:
        if name not in PUBLIC_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{name}'. Available: {', '.join(PUBLIC_BENCHMARKS)}"
            )
        loader = PUBLIC_BENCHMARKS[name]
        try:
            loaded = loader(samples_per_benchmark, seed)
            if not loaded:
                failures.append(name)
                logger.warning("No prompts loaded for public benchmark '%s'", name)
                continue
            prompts.extend(loaded)
            logger.info("Loaded %d prompts from public benchmark '%s'", len(loaded), name)
        except Exception as exc:
            failures.append(name)
            logger.warning("Failed to load public benchmark '%s': %s", name, exc)

    if failures:
        logger.warning("Skipped failed benchmarks: %s", ", ".join(failures))
    if not prompts:
        raise RuntimeError(
            f"Could not load any public benchmarks. Failures: {', '.join(failures)}"
        )

    return prompts
