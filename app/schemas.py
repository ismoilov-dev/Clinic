from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    faq = "faq"
    book_appointment = "book_appointment"
    cancel_appointment = "cancel_appointment"
    human_handoff = "human_handoff"
    other = "other"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    patient_name: str | None = None
    requested_date: str | None = None
    language: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@dataclass
class StreamEvent:
    type: str
    data: dict