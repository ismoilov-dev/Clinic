from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest, IntentResult
from app.services import classify, chat_stream


app = FastAPI(title="ClinicBot API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/classify", response_model=IntentResult)
async def classify_message(request: ChatRequest):
    return await classify(request.message)


@app.post("/chat/stream")
async def stream(request: ChatRequest):

    return StreamingResponse(
        chat_stream(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )