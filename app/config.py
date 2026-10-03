import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
    FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")
    PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.30"))
    PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.50"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "400"))


settings = Settings()