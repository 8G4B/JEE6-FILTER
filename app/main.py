import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks

from app.feedback import save_feedback, feedback_count
from app.inference import initialize_inference, predict_async, shutdown_inference
from app.metrics import metrics_app, observe_request

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_inference()
    logger.info("Profanity filter model loaded and warmed")
    try:
        yield
    finally:
        shutdown_inference()


app = FastAPI(title="JEE6 Profanity Filter", lifespan=lifespan)
app.middleware("http")(observe_request)
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_profanity(body: dict):
    text = body.get("text", "")
    if not text:
        return {"is_profanity": False, "confidence": 0.0}
    return await predict_async(text)


@app.post("/feedback")
async def add_feedback(body: dict):
    text = body.get("text", "")
    label = body.get("label", 0)
    if not text:
        return {"status": "error", "message": "text is required"}
    total = await asyncio.to_thread(save_feedback, text, label)
    return {"status": "ok", "total_feedback": total}


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    from app.trainer import train

    count = await asyncio.to_thread(feedback_count)
    if count < 5:
        return {"status": "skip", "message": f"피드백 {count}개 — 최소 5개 필요"}

    background_tasks.add_task(train)
    return {"status": "training", "samples": count}


@app.get("/status")
async def model_status():
    import os
    from app.model import FINE_TUNED_DIR

    return {
        "fine_tuned": os.path.exists(FINE_TUNED_DIR),
        "feedback_count": await asyncio.to_thread(feedback_count),
    }
