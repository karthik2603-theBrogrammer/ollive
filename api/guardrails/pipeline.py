from __future__ import annotations

from dataclasses import dataclass, field

from api.settings import GuardrailConfig
from api.guardrails.injection_guard import check_injection_and_limits
from api.guardrails.input_safety import check_input_safety
from api.guardrails.output_safety import REFUSAL_TEMPLATE, sanitize_output


@dataclass
class GuardrailDecision:
    allowed: bool
    response_text: str = ""
    blocked_layers: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


class GuardrailPipeline:
    def __init__(self, config: GuardrailConfig) -> None:
        self.config = config

    def check_input(self, text: str) -> GuardrailDecision:
        checks = [
            check_input_safety(text),
            check_injection_and_limits(text, max_chars=self.config.max_input_chars),
        ]
        blocked = [check for check in checks if not check.allowed]
        if blocked and self.config.block_on_input_violation:
            return GuardrailDecision(
                allowed=False,
                response_text=REFUSAL_TEMPLATE,
                blocked_layers=[item.layer for item in blocked],
                reasons=[item.reason for item in blocked if item.reason],
            )
        return GuardrailDecision(allowed=True)

    def check_output(self, text: str) -> GuardrailDecision:
        result = sanitize_output(text, guard_identity=True)
        if not result.allowed and self.config.block_on_output_violation:
            return GuardrailDecision(
                allowed=False,
                response_text=result.sanitized_text or REFUSAL_TEMPLATE,
                blocked_layers=[result.layer],
                reasons=[result.reason] if result.reason else [],
            )
        return GuardrailDecision(allowed=True, response_text=result.sanitized_text or text)
