from dataclasses import dataclass


@dataclass
class CallMetrics:
    endpoint: str
    model: str
    attempts: int
    success: bool
    ttft_ms: float
    total_ms: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int

    @property
    def cost_usd(self) -> float:
        return 0.0