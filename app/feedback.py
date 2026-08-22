import json
import logging
import os
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

FEEDBACK_FILE = "/data/feedback.jsonl"
_feedback_lock = threading.RLock()
_feedback_total: int | None = None


def save_feedback(text: str, correct_label: int) -> int:
    global _feedback_total

    entry = {
        "text": text,
        "label": correct_label,
        "timestamp": datetime.now().isoformat(),
    }
    with _feedback_lock:
        if _feedback_total is None:
            _feedback_total = _count_feedback_unlocked()
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _feedback_total += 1
    logger.info(f"피드백 저장: label={correct_label}, text={text[:30]}...")
    return _feedback_total


def load_feedback() -> list[dict]:
    with _feedback_lock:
        if not os.path.exists(FEEDBACK_FILE):
            return []
        entries = []
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries


def _count_feedback_unlocked() -> int:
    if not os.path.exists(FEEDBACK_FILE):
        return 0
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def feedback_count() -> int:
    global _feedback_total

    with _feedback_lock:
        if _feedback_total is None:
            _feedback_total = _count_feedback_unlocked()
        return _feedback_total
