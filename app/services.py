from app.llm_client import LLMClient
from app.prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    ASSISTANT_SYSTEM_PROMPT,
    build_classifier_prompt,
    build_assistant_prompt,
)
from app.schemas import IntentResult, StreamEvent


llm = LLMClient()


async def classify(message: str) -> IntentResult:
    """
    User message'ni classify qiladi.
    """

    # User message uchun classifier prompt yaratamiz
    prompt = build_classifier_prompt(message)

    # Gemini'ga yuboramiz
    result = await llm.generate_json(
        prompt=prompt,
        system=CLASSIFIER_SYSTEM_PROMPT,
        schema=IntentResult,
    )

    return result


async def chat_stream(message: str):
    """
    User savoliga Gemini'dan streaming javob oladi.
    """
    prompt = build_assistant_prompt(message)
    # Userning savolini Gemini'ga prompt sifatida beramiz
    async for event in llm.stream(
        prompt=prompt,
        system=ASSISTANT_SYSTEM_PROMPT,
    ):
        yield event