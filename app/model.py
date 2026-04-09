import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "kdyeon0309/gogo_forpanity_filter"

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()


def predict(text: str) -> dict:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax(dim=-1).item()
    confidence = probs[0][pred].item()

    return {
        "is_profanity": pred == 1,
        "confidence": round(confidence, 4),
    }
