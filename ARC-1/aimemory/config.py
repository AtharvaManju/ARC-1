from dataclasses import dataclass

@dataclass
class AIMemoryConfig:
    pool_dir: str = "/mnt/nvme_pool"
    backend: str = "AUTO"  # AUTO | NVME_FILE | RAM | NOOP
    durable: bool = False
    staging_mb: int = 512
    strict_direct: bool = False

    spill_min_bytes: int = 128 * 1024 * 1024
    io_workers: int = 2
    max_queue: int = 4096
    queue_put_timeout_s: float = 0.01
    spill_queue_overflow_policy: str = "SYNC_SPILL"  # BLOCK | SYNC_SPILL | DISABLE_SPILLING
    pack_wait_for_commit: bool = False

    pinned_pool_bytes: int = 2 * 1024**3
    prefetch_stream_priority: int = -1

    # PCC (unchanged logic; only config)
    pcc_lookahead: int = 4
    pcc_drift_disable_threshold: int = 64

    overhead_sla_pct: float = 12.0
    baseline_ema_alpha: float = 0.08
    safe_mode_cooldown_steps: int = 50

    encrypt_at_rest: bool = False
    encryption_key_path: str = ""

    telemetry_opt_in: bool = False
    offline_mode: bool = True
    telemetry_dir: str = "./aimemory_telemetry"
    enable_jsonl_logs: bool = True

    require_license: bool = False
    license_path: str = "/etc/aimemory/license.json"

    hard_fail_on_corruption: bool = True

    rank: int = -1
    world_size: int = -1
    warn_on_multinode: bool = True

    # Disk budget + GC
    max_pool_bytes: int = 500 * 1024**3
    gc_keep_last_windows: int = 2
    pool_window_steps: int = 50

    # NEW: deterministic retention cadence + DB maintenance
    gc_every_steps: int = 50                 # run retention cleanup every N steps (default: each window)
    gc_checkpoint_truncate: bool = True      # PRAGMA wal_checkpoint(TRUNCATE) after GC
    gc_vacuum: bool = False                  # expensive; off by default

    # RAM backend budget
    ram_max_bytes: int = 64 * 1024**3

    # NEW: perf killer toggle (default OFF)
    sync_each_step: bool = False             # if True, do torch.cuda.synchronize() each step end (debug/bench)
