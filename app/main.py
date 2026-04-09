import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from app.model import load_model, predict
from app.feedback import save_feedback, feedback_count

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Profanity filter model loaded")
    yield


app = FastAPI(title="JEE6 Profanity Filter", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_profanity(body: dict):
    text = body.get("text", "")
    if not text:
        return {"is_profanity": False, "confidence": 0.0}
    return predict(text)


@app.post("/feedback")
async def add_feedback(body: dict):
    text = body.get("text", "")
    label = body.get("label", 0)
    if not text:
        return {"status": "error", "message": "text is required"}
    save_feedback(text, label)
    return {"status": "ok", "total_feedback": feedback_count()}


@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    from app.trainer import train

    count = feedback_count()
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
        "feedback_count": feedback_count(),
    }
