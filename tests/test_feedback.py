from concurrent.futures import ThreadPoolExecutor

from app import feedback


def test_feedback_count_is_updated_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", str(tmp_path / "feedback.jsonl"))
    monkeypatch.setattr(feedback, "_feedback_total", None)

    with ThreadPoolExecutor(max_workers=4) as executor:
        totals = list(
            executor.map(
                lambda index: feedback.save_feedback(f"message-{index}", index % 2),
                range(20),
            )
        )

    assert sorted(totals) == list(range(1, 21))
    assert feedback.feedback_count() == 20
    assert len(feedback.load_feedback()) == 20
