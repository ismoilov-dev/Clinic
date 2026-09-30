from google import genai
from google.genai import types
from typing import AsyncIterator

from app.config import settings
from app.schemas import StreamEvent

class LLMClient:

    def __init__(self, client=None):
        self.client = client or genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

    async def generate_json(
        self,
        prompt: str,
        system: str,
        schema,
    ):
        response = self.client.models.generate_content(
            model=settings.PRIMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            ),
        )

        return schema.model_validate_json(response.text)
    
    
    # Stream
    async def stream(self,prompt: str,system: str,) -> AsyncIterator[StreamEvent]:
        """
        Gemini'dan javobni streaming ko'rinishida oladi.

        Ya'ni Gemini butun javobni birdan bermaydi.
        Kelgan har bir text chunkni alohida yuboramiz.
        """

    # Gemini'ga streaming request yuboramiz
        response = self.client.models.generate_content_stream(
            model=settings.PRIMARY_MODEL,

            # Userning prompti
            contents=prompt,

            # System prompt:
            # AI qanday ishlashi kerakligini aytadi
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            ),
        )

    # Gemini'dan kelayotgan har bir chunkni olamiz
        for chunk in response:

            # Ba'zi chunklarda text bo'lmasligi mumkin
            if not chunk.text:
                continue

            # Har bir text chunkni "token" event qilib yuboramiz
            yield StreamEvent(
                type="token",
                data={
                    "text": chunk.text
                }
            )