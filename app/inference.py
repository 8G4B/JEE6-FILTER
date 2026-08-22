import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from app.metrics import INFERENCE_DURATION
from app.model import load_model, predict


_executor: ThreadPoolExecutor | None = None


def initialize_inference() -> None:
    global _executor

    workers = max(1, int(os.getenv("FILTER_INFERENCE_WORKERS", "1")))
    torch_threads = max(1, int(os.getenv("FILTER_TORCH_THREADS", "2")))
    torch.set_num_threads(torch_threads)

    load_model()
    predict("모델 준비")
    _executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="filter-inference",
    )


async def predict_async(text: str) -> dict:
    if _executor is None:
        raise RuntimeError("inference executor is not initialized")

    started_at = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_executor, predict, text)
    finally:
        INFERENCE_DURATION.observe(time.perf_counter() - started_at)


def shutdown_inference() -> None:
    global _executor

    if _executor is not None:
        _executor.shutdown(wait=True, cancel_futures=True)
    _executor = None
