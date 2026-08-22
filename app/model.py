import os
import logging
import threading

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "kdyeon0309/gogo_forpanity_filter"
TOKENIZER_NAME = "beomi/KcELECTRA-base"
FINE_TUNED_DIR = "/data/fine_tuned_model"

_tokenizer = None
_model = None
_model_lock = threading.RLock()


def load_model():
    global _tokenizer, _model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if os.path.exists(FINE_TUNED_DIR):
        logger.info(f"Fine-tuned 모델 로드: {FINE_TUNED_DIR}")
        model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_DIR)
    else:
        logger.info(f"베이스 모델 로드: {BASE_MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)

    model.eval()
    if os.getenv("FILTER_DYNAMIC_QUANTIZATION", "false").lower() in (
        "true",
        "1",
        "yes",
    ):
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("필터 동적 INT8 양자화 적용")

    with _model_lock:
        _tokenizer = tokenizer
        _model = model


def predict(text: str) -> dict:
    with _model_lock:
        tokenizer = _tokenizer
        model = _model
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.inference_mode():
            outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax(dim=-1).item()
    confidence = probs[0][pred].item()

    # 3-class: 0=clean, 1=mild profanity, 2=severe profanity
    return {
        "is_profanity": pred >= 1,
        "label": pred,
        "confidence": round(confidence, 4),
    }


def get_model_and_tokenizer():
    return _model, _tokenizer
