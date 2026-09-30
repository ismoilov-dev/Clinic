from dataclasses import dataclass


@dataclass
class CallMetrics:
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: float