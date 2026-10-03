"""
All settings live here, read from the .env file.

Why: prices, model names and limits change often. Keeping them in .env
means you change behaviour without touching code (like Django's settings.py).
"""

import os

from dotenv import load_dotenv

# Read the .env file and put its values into os.environ
load_dotenv()


def _optional_int(name: str) -> int | None:
    """Return an int from .env, or None if the variable is empty/missing."""
    value = os.getenv(name, "").strip()
    return int(value) if value else None


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # ⚠️ Check current model names: https://ai.google.dev/gemini-api/docs/models
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")

    # ⚠️ Placeholder prices in USD per 1 million tokens.
    # Copy real values for YOUR model from https://ai.google.dev/pricing
    PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.30"))
    PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.50"))

    # How many times we try the PRIMARY model before falling back
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # Hard cap on answer length: shorter answers = cheaper + faster (Day 13)
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "400"))

    # Optional. Empty = let the model decide.
    # 0 = turn thinking off (works on Gemini 2.5 Flash; other models may differ).
    THINKING_BUDGET = _optional_int("THINKING_BUDGET")


# One shared instance, imported everywhere: `from app.config import settings`
settings = Settings()