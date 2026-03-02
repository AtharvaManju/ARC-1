from typing import Dict, Any

import torch


def allocator_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda": False,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "inactive_split_bytes": 0,
            "fragmentation_pct": 0.0,
            "reclaimable_bytes": 0,
            "segments": 0,
        }
    try:
        stats = torch.cuda.memory_stats()
    except Exception:
        stats = {}
    allocated = int(stats.get("allocated_bytes.all.current", torch.cuda.memory_allocated()))
    reserved = int(stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved()))
    inactive_split = int(stats.get("inactive_split_bytes.all.current", 0))
    segments = int(stats.get("segment.all.current", 0))
    reclaim = max(0, reserved - allocated)
    frag = (100.0 * float(inactive_split) / float(max(1, reserved))) if reserved > 0 else 0.0
    return {
        "cuda": True,
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "inactive_split_bytes": inactive_split,
        "fragmentation_pct": float(frag),
        "reclaimable_bytes": int(reclaim),
        "segments": int(segments),
    }
