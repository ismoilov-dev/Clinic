"""
Data shapes (contracts) used across the whole app.

Pydantic models = the same idea as DRF serializers: they validate data
coming IN (from users, and from the LLM!) before we trust it.
"""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What the patient wants. `str` base = serializes to plain JSON strings."""

    faq = "faq"
    book_appointment = "book_appointment"
    cancel_appointment = "cancel_appointment"
    human_handoff = "human_handoff"
    other = "other"


class IntentResult(BaseModel):
    """What the classifier LLM must return. Gemini gets this as response_schema."""

    intent: Intent
    # ge/le = "greater or equal" / "less or equal": 1.5 will be REJECTED
    confidence: float = Field(ge=0, le=1)
    # None = the user did not say it. We never want the model to invent these.
    patient_name: str | None = None
    requested_date: str | None = None
    language: str  # "uz", "ru", "en" (or "unknown" in our safe fallback)


class ChatRequest(BaseModel):
    """Request body for /classify and /chat/stream."""

    message: str = Field(min_length=1, max_length=2000)


@dataclass
class StreamEvent:
    """
    One item coming out of LLMClient.stream().

    type = "token" → data = {"text": "..."}
    type = "final" → data = {"metrics": CallMetrics}
    """

    type: str
    data: dict = field(default_factory=dict)