import os
import sqlite3
import glob
from typing import Optional

def _rank_dir(pool_dir: str, rank: int) -> str:
    return os.path.join(pool_dir, f"rank_{rank}")

def _db_path(pool_dir: str, rank: int) -> str:
    return os.path.join(_rank_dir(pool_dir, rank), "metadata.db")

def _file_size(p: str) -> int:
    try:
        return os.path.getsize(p)
    except Exception:
        return 0

def _total_pool_bytes(rank_dir: str) -> int:
    total = 0
    for p in glob.glob(os.path.join(rank_dir, "pool_*.bin")):
        total += _file_size(p)
    return total

def gc_old_windows(pool_dir: str, rank: int, cutoff_pool_id: int, checkpoint_truncate: bool, vacuum: bool) -> dict:
    rank_dir = _rank_dir(pool_dir, rank)
    db = _db_path(pool_dir, rank)
    if not os.path.exists(db):
        return {"ok": True, "reason": "no db"}

    conn = sqlite3.connect(db, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")

    cur = conn.execute("DELETE FROM tensors WHERE committed=1 AND pool_id < ?", (int(cutoff_pool_id),))
    deleted_rows = cur.rowcount
    conn.commit()

    if checkpoint_truncate:
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.commit()
        except Exception:
            pass

    if vacuum:
        # Expensive — opt-in only
        try:
            conn.execute("VACUUM;")
            conn.commit()
        except Exception:
            pass

    conn.close()

    deleted_files = 0
    for p in glob.glob(os.path.join(rank_dir, "pool_*.bin")):
        base = os.path.basename(p)
        try:
            pid = int(base.split("_")[1].split(".")[0])
        except Exception:
            continue
        if pid < int(cutoff_pool_id):
            try:
                os.remove(p)
                deleted_files += 1
            except Exception:
                pass

    return {"ok": True, "deleted_rows": deleted_rows, "deleted_files": deleted_files, "cutoff_pool_id": int(cutoff_pool_id)}

def gc_windows(
    pool_dir: str,
    rank: int,
    current_step: int,
    window_steps: int,
    keep_last_windows: int,
    checkpoint_truncate: bool,
    vacuum: bool,
) -> dict:
    cur_pool_id = int(current_step) // max(1, int(window_steps))
    cutoff_pool_id = max(0, cur_pool_id - int(keep_last_windows) + 1)
    return gc_old_windows(pool_dir, rank, cutoff_pool_id, checkpoint_truncate=checkpoint_truncate, vacuum=vacuum)

def gc_windows_if_needed(
    pool_dir: str,
    rank: int,
    current_step: int,
    window_steps: int,
    keep_last_windows: int,
    max_pool_bytes: int,
    every_steps: int,
    checkpoint_truncate: bool,
    vacuum: bool,
) -> Optional[dict]:
    """
    Sellable behavior:
      - ALWAYS enforce window retention periodically (every_steps).
      - max_pool_bytes remains an emergency trigger (can force GC earlier).
    """
    rank_dir = _rank_dir(pool_dir, rank)
    os.makedirs(rank_dir, exist_ok=True)

    do_periodic = (int(every_steps) > 0) and (int(current_step) % int(every_steps) == 0)
    over_budget = _total_pool_bytes(rank_dir) > int(max_pool_bytes)

    if not do_periodic and not over_budget:
        return None

    return gc_windows(
        pool_dir=pool_dir,
        rank=rank,
        current_step=current_step,
        window_steps=window_steps,
        keep_last_windows=keep_last_windows,
        checkpoint_truncate=checkpoint_truncate,
        vacuum=vacuum,
    )
