"""
Measuring every LLM call: tokens, cost, latency (Day 13).

For now records live in memory (a Python list). They disappear when the
server restarts. On Day 39 this becomes a PostgreSQL table.
"""

import statistics
from dataclasses import dataclass

from app.config import settings


@dataclass
class CallMetrics:
    """Everything we know about ONE LLM call."""

    model: str               # which model actually answered
    attempts: int            # total tries (retries + fallback included)
    success: bool = True
    ttft_ms: float = 0.0     # time to first token (for JSON calls = total)
    total_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    endpoint: str = ""       # "classify" or "chat" - set by services.py

    @property
    def cost_usd(self) -> float:
        """Prices are per 1M tokens. Thinking tokens are billed as OUTPUT."""
        billed_output = self.output_tokens + self.thinking_tokens
        return (
            self.input_tokens * settings.PRICE_INPUT_PER_1M
            + billed_output * settings.PRICE_OUTPUT_PER_1M
        ) / 1_000_000

    def to_dict(self) -> dict:
        """For JSON responses / SSE events."""
        return {
            "model": self.model,
            "attempts": self.attempts,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


def usage_to_tokens(usage) -> tuple[int, int, int]:
    """
    Read token counts from Gemini's usage_metadata.
    Any field can be None, so `or 0` turns None into 0.
    """
    if usage is None:
        return 0, 0, 0
    return (
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
    )


def _p50_p95(values: list[float]) -> tuple[float | None, float | None]:
    """Median and 95th percentile. quantiles() needs at least 2 values."""
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    p50 = statistics.median(values)
    # n=20 splits data into 20 parts -> index 18 is the 95% cut point
    p95 = statistics.quantiles(values, n=20)[18]
    return round(p50, 1), round(p95, 1)


class MetricsStore:
    """Keeps all CallMetrics and summarizes them for GET /metrics."""

    def __init__(self):
        self.records: list[CallMetrics] = []

    def record(self, m: CallMetrics) -> None:
        self.records.append(m)

    def summary(self) -> dict:
        ok = [r for r in self.records if r.success]
        ttft_p50, ttft_p95 = _p50_p95([r.ttft_ms for r in ok])
        total_p50, total_p95 = _p50_p95([r.total_ms for r in ok])

        return {
            "total_calls": len(self.records),
            "errors": len(self.records) - len(ok),
            # a fallback = some model other than the primary answered
            "fallbacks": sum(1 for r in ok if r.model != settings.PRIMARY_MODEL),
            "total_cost_usd": round(sum(r.cost_usd for r in self.records), 6),
            "ttft_ms": {"p50": ttft_p50, "p95": ttft_p95},
            "total_ms": {"p50": total_p50, "p95": total_p95},
        }


# One shared store for the whole app
metrics_store = MetricsStore()