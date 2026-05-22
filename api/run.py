#!/usr/bin/env python3
"""Run the ollive public OSS FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from api.settings import ApiConfig


def main() -> None:
    import os

    config = ApiConfig()
    port = int(os.getenv("PORT", str(config.port)))
    uvicorn.run(
        "api.main:app",
        host=config.host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
