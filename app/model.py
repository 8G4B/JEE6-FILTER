import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "kdyeon0309/gogo_forpanity_filter"
TOKENIZER_NAME = "beomi/KcELECTRA-base"
FINE_TUNED_DIR = "/data/fine_tuned_model"

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if os.path.exists(FINE_TUNED_DIR):
        logger.info(f"Fine-tuned 모델 로드: {FINE_TUNED_DIR}")
        _model = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_DIR)
    else:
        logger.info(f"베이스 모델 로드: {BASE_MODEL_NAME}")
        _model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)

    _model.eval()


def predict(text: str) -> dict:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _model(**inputs)

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
