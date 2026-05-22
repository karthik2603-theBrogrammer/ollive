import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    service_name: str = os.getenv("OTEL_SERVICE_NAME", "ollive-oss-api")
    max_spans: int = int(os.getenv("OTEL_MAX_SPANS", "2000"))


@dataclass(frozen=True)
class GuardrailConfig:
    max_input_chars: int = int(os.getenv("GUARDRAIL_MAX_INPUT_CHARS", "4000"))
    block_on_input_violation: bool = True
    block_on_output_violation: bool = True


@dataclass(frozen=True)
class CostConfig:
    cpu_hour_usd: float = float(os.getenv("API_CPU_HOUR_USD", "0.042"))
    estimated_tokens_per_char: float = float(os.getenv("API_TOKENS_PER_CHAR", "0.25"))


@dataclass(frozen=True)
class ApiConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    api_key: str = os.getenv("API_SERVICE_KEY", "")
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))
    benchmark_samples: int = int(os.getenv("BENCHMARK_SAMPLES", "10"))
    benchmark_seed: int = int(os.getenv("BENCHMARK_SEED", "42"))
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    cost: CostConfig = field(default_factory=CostConfig)
