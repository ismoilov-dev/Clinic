"""
Business logic. Endpoints in main.py call these functions.

Same layering as Django/DRF: views stay thin, logic lives in services.
"""

import json
import logging

from app.config import settings
from app.llm_client import LLMClient, LLMError
from app.metrics import CallMetrics, metrics_store
from app.prompts import (
    ASSISTANT_SYSTEM_PROMPT,
    CLASSIFIER_SYSTEM_PROMPT,
    build_assistant_prompt,
    build_classifier_prompt,
)
from app.schemas import Intent, IntentResult

logger = logging.getLogger("clinicbot.services")

llm = LLMClient()


# ------------------------------------------------------------
# /classify
# ------------------------------------------------------------
async def classify(message: str) -> IntentResult:
    try:
        result, metrics = await llm.generate_json(
            prompt=build_classifier_prompt(message),
            system=CLASSIFIER_SYSTEM_PROMPT,
            schema=IntentResult,
        )
        metrics.endpoint = "classify"
        metrics_store.record(metrics)
        return result

    except LLMError as e:
        # The real reason goes to OUR logs...
        logger.exception("classify failed")
        metrics_store.record(
            CallMetrics(
                model=settings.PRIMARY_MODEL,
                attempts=e.attempts,
                success=False,
                endpoint="classify",
            )
        )
        # ...and the user gets a SAFE default instead of a 500 error.
        # A receptionist that says "let me connect you to staff"
        # is better than one that crashes.
        return IntentResult(
            intent=Intent.human_handoff,
            confidence=0.0,
            language="unknown",
        )


# ------------------------------------------------------------
# /chat/stream
# ------------------------------------------------------------
def sse(event: str, data: dict) -> str:
    """
    Turn one event into Server-Sent Events text.
    StreamingResponse can only send str/bytes, NOT Python objects.
    json.dumps escapes newlines, so a long answer can't break the format.
    Blank line at the end = "this event is finished".
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def chat_stream(message: str):
    """Async generator of SSE strings: token... token... done (or error)."""
    try:
        async for event in llm.stream(
            prompt=build_assistant_prompt(message),
            system=ASSISTANT_SYSTEM_PROMPT,
        ):
            if event.type == "token":
                yield sse("token", event.data)

            elif event.type == "final":
                metrics: CallMetrics = event.data["metrics"]
                metrics.endpoint = "chat"
                metrics_store.record(metrics)
                yield sse("done", metrics.to_dict())

    except Exception as e:
        # HTTP 200 was already sent with the first byte, so we can't
        # return a 500 now. Instead we send an "error" EVENT (Day 12).
        logger.exception("chat stream failed")
        metrics_store.record(
            CallMetrics(
                model=settings.PRIMARY_MODEL,
                attempts=getattr(e, "attempts", 1),
                success=False,
                endpoint="chat",
            )
        )
        # Generic message only: never leak internal error text to users
        yield sse("error", {"message": "LLM request failed", "type": type(e).__name__})