"""Central logging setup for CLI and app entrypoints."""

from __future__ import annotations

import logging
import os
import warnings

ALLOWED_LEVELS = {
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

QUIET_LOGGER_NAMES = (
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "datasets",
    "huggingface_hub",
    "filelock",
    "gradio",
    "openai",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "evaluation",
    "assistants",
    "llm",
    "memory",
)


def configure_logging(level: str | None = None, *, fmt: str | None = None) -> None:
    """Configure app logging to show only WARNING and ERROR by default."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    requested = (level or os.getenv("LOG_LEVEL", "WARNING")).upper()
    log_level = ALLOWED_LEVELS.get(requested, logging.WARNING)

    logging.basicConfig(
        level=log_level,
        format=fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(log_level)

    for name in QUIET_LOGGER_NAMES:
        logging.getLogger(name).setLevel(log_level)

    try:
        from datasets.utils.logging import set_verbosity_warning

        set_verbosity_warning()
        import datasets

        datasets.disable_progress_bars()
    except ImportError:
        pass

    try:
        from transformers.utils.logging import set_verbosity_warning

        set_verbosity_warning()
    except ImportError:
        pass

    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    try:
        from langchain_core._api.deprecation import LangChainDeprecationWarning

        warnings.filterwarnings(
            "ignore",
            category=LangChainDeprecationWarning,
            message=r".*\.text\(\).*",
        )
    except ImportError:
        pass
