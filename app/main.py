import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.model import load_model, predict

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

    result = predict(text)
    return result
