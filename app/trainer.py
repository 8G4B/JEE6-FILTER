import logging
import threading

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification

from app.feedback import load_feedback
from app.model import (
    BASE_MODEL_NAME,
    FINE_TUNED_DIR,
    load_model,
    get_model_and_tokenizer,
)

logger = logging.getLogger(__name__)

EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-5
_training_lock = threading.Lock()
_training = False


def claim_training() -> bool:
    global _training

    with _training_lock:
        if _training:
            return False
        _training = True
        return True


def is_training() -> bool:
    with _training_lock:
        return _training


def _finish_training() -> None:
    global _training

    with _training_lock:
        _training = False


class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train(*, claimed: bool = False) -> dict:
    if not claimed and not claim_training():
        return {"status": "skip", "message": "이미 학습 중입니다."}

    try:
        return _train()
    finally:
        _finish_training()


def _train() -> dict:
    feedback = load_feedback()
    if len(feedback) < 5:
        return {
            "status": "skip",
            "message": f"피드백 {len(feedback)}개 — 최소 5개 필요",
        }

    texts = [f["text"] for f in feedback]
    labels = [f["label"] for f in feedback]

    _, tokenizer = get_model_and_tokenizer()

    # Always start from base model to avoid catastrophic forgetting
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)
    model.train()

    dataset = FeedbackDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_loss = 0
    steps = 0

    for epoch in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / steps if steps > 0 else 0

    model.save_pretrained(FINE_TUNED_DIR)
    logger.info(f"Fine-tuned 모델 저장 완료: {FINE_TUNED_DIR}")

    # Reload the new model into memory
    load_model()

    return {
        "status": "ok",
        "samples": len(feedback),
        "epochs": EPOCHS,
        "avg_loss": round(avg_loss, 4),
    }
