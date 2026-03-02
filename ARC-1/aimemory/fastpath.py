from typing import Dict, Any


def choose_decompress_path(codec: str, nbytes: int, mode: str = "auto", gpu_available: bool = False, gpu_min_bytes: int = 8 * 1024 * 1024) -> str:
    m = str(mode or "auto").lower()
    if str(codec).lower() in ("none", ""):
        return "none"
    if m == "cpu":
        return "cpu"
    if m == "gpu":
        return "gpu" if gpu_available else "cpu_fallback"
    if gpu_available and int(nbytes) >= int(max(1, gpu_min_bytes)):
        # Placeholder for nvCOMP-like path when available.
        return "gpu_candidate"
    return "cpu"


def can_direct_restore(meta: Dict[str, Any], gds_enabled: bool, direct_restore_enabled: bool) -> bool:
    if not bool(direct_restore_enabled):
        return False
    if not bool(gds_enabled):
        return False
    if int(meta.get("encrypted", 0)) != 0:
        return False
    if int(meta.get("compressed", 0)) != 0:
        return False
    return True
