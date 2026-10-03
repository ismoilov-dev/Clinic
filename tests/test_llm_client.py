"""
Deterministic tests: NO real API calls, NO network, NO money spent.

We build a FAKE Gemini client whose answers we script, then check that
LLMClient reacts correctly (retry, fallback, give up).
Same idea as mocking the database in backend tests.

Run: pytest tests/test_llm_client.py -v
"""

from types import SimpleNamespace

import pytest
from google.genai import errors

from app.config import settings
from app.llm_client import LLMClient, LLMError
from app.schemas import Intent, IntentResult

VALID_JSON = '{"intent": "faq", "confidence": 0.9, "patient_name": null, "requested_date": null, "language": "en"}'
USAGE = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=0)


# ------------------------------------------------------------
# Fake Gemini client
# ------------------------------------------------------------
def api_error(code: int) -> errors.APIError:
    """Build a real SDK error object with the given HTTP code."""
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": "fake", "status": "FAKE"}})


def ok_response(text: str = VALID_JSON):
    return SimpleNamespace(text=text, usage_metadata=USAGE)


class FakeModels:
    """
    `script` = list of things to return, one per call, in order.
    An Exception in the list is RAISED instead of returned.
    Every call's model name is saved in `calls`, so tests can check it.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[str] = []

    def _next(self, model):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def generate_content(self, model, contents, config):
        return self._next(model)

    async def generate_content_stream(self, model, contents, config):
        chunks = self._next(model)          # a list of chunk objects

        async def gen():
            for c in chunks:
                if isinstance(c, Exception):
                    raise c
                yield c

        return gen()


def make_client(script):
    fake_models = FakeModels(script)
    fake = SimpleNamespace(aio=SimpleNamespace(models=fake_models))

    async def no_sleep(_seconds):           # don't really wait in tests
        pass

    return LLMClient(client=fake, sleep=no_sleep), fake_models


def call_json(llm):
    return llm.generate_json(prompt="hi", system="sys", schema=IntentResult)


# ------------------------------------------------------------
# generate_json
# ------------------------------------------------------------
async def test_happy_path():
    llm, fake = make_client([ok_response()])
    result, m = await call_json(llm)
    assert result.intent == Intent.faq
    assert m.attempts == 1
    assert m.input_tokens == 100 and m.cost_usd > 0


async def test_invalid_json_is_retried():
    llm, fake = make_client([ok_response("not json at all"), ok_response()])
    result, m = await call_json(llm)
    assert m.attempts == 2


async def test_429_is_retried():
    llm, fake = make_client([api_error(429), ok_response()])
    result, m = await call_json(llm)
    assert m.attempts == 2
    assert m.model == settings.PRIMARY_MODEL


async def test_401_is_not_retried():
    llm, fake = make_client([api_error(401), ok_response()])
    with pytest.raises(LLMError) as exc:
        await call_json(llm)
    assert exc.value.attempts == 1
    assert len(fake.calls) == 1              # really only ONE call was made


async def test_fallback_after_primary_fails():
    failures = [api_error(503)] * settings.MAX_RETRIES
    llm, fake = make_client(failures + [ok_response()])
    result, m = await call_json(llm)
    assert m.model == settings.FALLBACK_MODEL
    assert m.attempts == settings.MAX_RETRIES + 1


async def test_404_jumps_straight_to_fallback():
    llm, fake = make_client([api_error(404), ok_response()])
    result, m = await call_json(llm)
    assert fake.calls == [settings.PRIMARY_MODEL, settings.FALLBACK_MODEL]


# ------------------------------------------------------------
# stream
# ------------------------------------------------------------
def chunk(text, usage=None):
    return SimpleNamespace(text=text, usage_metadata=usage)


async def collect(llm):
    return [e async for e in llm.stream(prompt="hi", system="sys")]


async def test_stream_tokens_then_final():
    llm, _ = make_client([[chunk("Hello "), chunk("there", USAGE)]])
    events = await collect(llm)
    assert [e.type for e in events] == ["token", "token", "final"]
    assert events[-1].data["metrics"].output_tokens == 20


async def test_stream_retries_before_first_token():
    llm, _ = make_client([api_error(429), [chunk("Hi", USAGE)]])
    events = await collect(llm)
    assert events[-1].data["metrics"].attempts == 2


async def test_stream_does_not_retry_after_first_token():
    # the stream breaks AFTER "Hello" was already sent to the user
    llm, fake = make_client([[chunk("Hello "), api_error(503)], [chunk("second try")]])
    with pytest.raises(LLMError):
        await collect(llm)
    assert len(fake.calls) == 1              # no second attempt