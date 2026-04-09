import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

FEEDBACK_FILE = "/data/feedback.jsonl"


def save_feedback(text: str, correct_label: int):
    entry = {
        "text": text,
        "label": correct_label,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"피드백 저장: label={correct_label}, text={text[:30]}...")


def load_feedback() -> list[dict]:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def feedback_count() -> int:
    if not os.path.exists(FEEDBACK_FILE):
        return 0
    with open(FEEDBACK_FILE, "r") as f:
        return sum(1 for line in f if line.strip())
