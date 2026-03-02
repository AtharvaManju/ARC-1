import json
import os
import time
from typing import Dict, Any

def emit_telemetry(event: Dict[str, Any], enabled: bool, offline_mode: bool, out_dir: str):
    if not enabled:
        return
    os.makedirs(out_dir, exist_ok=True)
    event = dict(event)
    event["ts"] = time.time()
    path = os.path.join(out_dir, "telemetry.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
