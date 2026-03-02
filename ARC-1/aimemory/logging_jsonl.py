import json
import os
import time
from typing import Any, Dict

class JsonlLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, event: Dict[str, Any]):
        event = dict(event)
        event.setdefault("ts", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

def default_rank_log_path(pool_dir: str, rank: int) -> str:
    return os.path.join(pool_dir, f"rank_{rank}", "aimemory.log.jsonl")

def safe_exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"
