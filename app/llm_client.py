import asyncio
import logging
import random
import time
from typing import AsyncIterator

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.metrics import CallMetrics, usage_to_tokens
from app.schemas import StreamEvent

logger = logging.getLogger("clinicbot.llm")

# Status codes where waiting and trying again can actually help
RETRYABLE_CODES = {429, 500, 502, 503, 504}


class LLMError(Exception):

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts


def backoff_delay(attempt: int, base: float = 1.0) -> float:

    return base * (2 ** (attempt - 1)) + random.uniform(0, 0.5)


class LLMClient:
    def __init__(self, client=None, sleep=asyncio.sleep):
        # Dependency injection: tests pass a FAKE client and a no-op sleep,
        # exactly like mocking the database in backend tests.
        self.client = client or genai.Client(api_key=settings.GEMINI_API_KEY)
        self._sleep = sleep

    # ------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------
    def _model_plan(self) -> list[tuple[str, int]]:
        """[(model, how many tries)] - primary N times, then fallback once."""
        plan = [(settings.PRIMARY_MODEL, settings.MAX_RETRIES)]
        if settings.FALLBACK_MODEL and settings.FALLBACK_MODEL != settings.PRIMARY_MODEL:
            plan.append((settings.FALLBACK_MODEL, 1))
        return plan

    def _config(self, system: str, **extra) -> types.GenerateContentConfig:
        """Shared config for every call."""
        kwargs = dict(
            system_instruction=system,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            **extra,
        )
        if settings.THINKING_BUDGET is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=settings.THINKING_BUDGET
            )
        return types.GenerateContentConfig(**kwargs)

    async def _handle_api_error(self, e: errors.APIError, model: str, attempt: int, total: int) -> str:
        """
        Decide what to do with an API error.
        Returns "retry" or "next_model", or raises LLMError to stop.
        """
        if e.code in RETRYABLE_CODES:
            delay = backoff_delay(attempt)
            logger.warning("%s: HTTP %s, retrying in %.1fs", model, e.code, delay)
            await self._sleep(delay)
            return "retry"
        if e.code == 404:
            # Model not found (typo in .env, model retired) -> try the fallback
            logger.error("%s: model not found (404), switching model", model)
            return "next_model"
        # 400 bad request, 401/403 bad key: retrying just wastes money
        raise LLMError(f"Non-retryable error {e.code}: {e}", total) from e

    # ------------------------------------------------------------
    # 1) Structured output: validation + retry + fallback
    # ------------------------------------------------------------
    async def generate_json(
        self, prompt: str, system: str, schema: type[BaseModel]
    ) -> tuple[BaseModel, CallMetrics]:
        total_attempts = 0
        last_error: Exception | None = None

        for model, max_tries in self._model_plan():
            for attempt in range(1, max_tries + 1):
                total_attempts += 1
                start = time.perf_counter()
                try:
                    # .aio = the ASYNC client -> doesn't block other users
                    response = await self.client.aio.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=self._config(
                            system,
                            response_mime_type="application/json",
                            response_schema=schema,
                        ),
                    )

                    # Never trust the model: validate the JSON ourselves
                    result = schema.model_validate_json(response.text or "")

                    total_ms = round((time.perf_counter() - start) * 1000, 1)
                    inp, out, think = usage_to_tokens(response.usage_metadata)
                    metrics = CallMetrics(
                        model=model,
                        attempts=total_attempts,
                        ttft_ms=total_ms,   # no streaming here: first token = whole answer
                        total_ms=total_ms,
                        input_tokens=inp,
                        output_tokens=out,
                        thinking_tokens=think,
                    )
                    return result, metrics

                except ValidationError as e:
                    # Bad JSON / wrong shape -> just ask again (no waiting needed)
                    logger.warning("%s: invalid JSON on attempt %s", model, attempt)
                    last_error = e

                except errors.APIError as e:
                    last_error = e
                    action = await self._handle_api_error(e, model, attempt, total_attempts)
                    if action == "next_model":
                        break  # leave the inner loop -> next model in the plan

        raise LLMError(f"All attempts failed. Last error: {last_error}", total_attempts)

    # ------------------------------------------------------------
    # 2) Streaming: retry/fallback ONLY before the first token
    # ------------------------------------------------------------
    async def stream(self, prompt: str, system: str) -> AsyncIterator[StreamEvent]:
        total_attempts = 0
        last_error: Exception | None = None

        for model, max_tries in self._model_plan():
            for attempt in range(1, max_tries + 1):
                total_attempts += 1
                start = time.perf_counter()
                ttft_ms = None
                usage = None

                try:
                    # first `await` opens the stream, then `async for` reads it
                    response = await self.client.aio.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=self._config(system),
                    )

                    async for chunk in response:
                        # usage_metadata comes with the chunks; keep the latest
                        if chunk.usage_metadata:
                            usage = chunk.usage_metadata
                        if not chunk.text:
                            continue
                        if ttft_ms is None:
                            ttft_ms = round((time.perf_counter() - start) * 1000, 1)
                        yield StreamEvent(type="token", data={"text": chunk.text})

                    # Stream finished -> send one final event with the numbers
                    total_ms = round((time.perf_counter() - start) * 1000, 1)
                    inp, out, think = usage_to_tokens(usage)
                    metrics = CallMetrics(
                        model=model,
                        attempts=total_attempts,
                        ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
                        total_ms=total_ms,
                        input_tokens=inp,
                        output_tokens=out,
                        thinking_tokens=think,
                    )
                    yield StreamEvent(type="final", data={"metrics": metrics})
                    return

                except errors.APIError as e:
                    # The user already SAW some text -> a retry would show the
                    # answer twice or change it. So we stop here (Day 12).
                    if ttft_ms is not None:
                        raise LLMError(f"Stream broke after first token: {e}", total_attempts) from e
                    last_error = e
                    action = await self._handle_api_error(e, model, attempt, total_attempts)
                    if action == "next_model":
                        break

        raise LLMError(f"All stream attempts failed. Last error: {last_error}", total_attempts)