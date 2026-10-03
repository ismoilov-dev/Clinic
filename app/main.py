"""
HTTP layer only. Every endpoint is a few lines: logic lives in services.py.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.metrics import metrics_store
from app.schemas import ChatRequest, IntentResult
from app.services import chat_stream, classify

# Show INFO/WARNING logs (retries, fallbacks) in the terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="ClinicBot API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify", response_model=IntentResult)
async def classify_message(request: ChatRequest):
    return await classify(request.message)


@app.post("/chat/stream")
async def chat(request: ChatRequest):
    return StreamingResponse(
        chat_stream(request.message),        # yields SSE strings
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # stop Nginx from buffering the stream
        },
    )


@app.get("/metrics")
async def metrics():
    return metrics_store.summary()