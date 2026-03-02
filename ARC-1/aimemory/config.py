from dataclasses import dataclass

@dataclass
class AIMemoryConfig:
    pool_dir: str = "/mnt/nvme_pool"
    backend: str = "AUTO"  # AUTO | NVME_FILE | RAM | NOOP
    durable: bool = False
    staging_mb: int = 512
    strict_direct: bool = False
    compile_safe_mode: bool = True
    strict_local_pool: bool = True

    spill_min_bytes: int = 128 * 1024 * 1024
    io_workers: int = 2
    max_queue: int = 4096
    queue_put_timeout_s: float = 0.01
    spill_queue_overflow_policy: str = "SYNC_SPILL"  # BLOCK | SYNC_SPILL | DISABLE_SPILLING
    pack_wait_for_commit: bool = False
    prefetch_batch_limit: int = 8
    dynamic_prefetch_limit_min: int = 1
    dynamic_prefetch_limit_max: int = 16
    queue_soft_limit_ratio: float = 0.85

    pinned_pool_bytes: int = 2 * 1024**3
    prefetch_stream_priority: int = -1
    inflight_spill_bytes_limit: int = 8 * 1024**3
    inflight_prefetch_bytes_limit: int = 8 * 1024**3
    per_step_spill_budget_bytes: int = 16 * 1024**3
    per_window_spill_budget_bytes: int = 128 * 1024**3
    spill_budget_window_steps: int = 50
    fairness_per_step_bytes: int = 8 * 1024**3
    nvme_write_bw_limit_mb_s: float = 0.0
    nvme_read_bw_limit_mb_s: float = 0.0

    # PCC (unchanged logic; only config)
    pcc_lookahead: int = 4
    pcc_drift_disable_threshold: int = 64

    overhead_sla_pct: float = 12.0
    baseline_ema_alpha: float = 0.08
    safe_mode_cooldown_steps: int = 50

    encrypt_at_rest: bool = False
    encryption_key_path: str = ""
    kms_key_uri: str = ""   # env://AIMEMORY_KEY_HEX, env://AIMEMORY_KEY_B64, file:///path/to/key
    compression_codec: str = "none"  # none | auto | zlib | lz4 | zstd
    compression_min_bytes: int = 4 * 1024 * 1024
    compression_min_gain_ratio: float = 0.05
    audit_log_path: str = ""
    enable_audit_log: bool = False

    telemetry_opt_in: bool = False
    offline_mode: bool = True
    telemetry_dir: str = "./aimemory_telemetry"
    enable_jsonl_logs: bool = True

    require_license: bool = False
    license_path: str = "/etc/aimemory/license.json"

    hard_fail_on_corruption: bool = True
    strict_read_barrier: bool = True
    stronger_payload_hash: bool = True
    quarantine_on_corruption: bool = True

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

    # Runtime autotuner
    autotune_enabled: bool = True
    autotune_profile_dir: str = ""
    autotune_adjust_interval_steps: int = 10
    model_profile_name: str = ""

    # Pressure governor (OOM-as-degrade)
    governor_enabled: bool = True
    governor_warn_headroom_pct: float = 12.0
    governor_emergency_headroom_pct: float = 6.0
    governor_cooldown_steps: int = 5
    governor_spill_min_floor_bytes: int = 1 * 1024 * 1024
    fail_open_on_error: bool = True
    unpack_meta_wait_timeout_s: float = 2.0
    unpack_meta_wait_poll_s: float = 0.002

    # Determinism / stream semantics
    deterministic_stream_fences: bool = True
    deterministic_io_ordering: bool = True
    collective_safe_mode: bool = True
    collective_cadence_steps: int = 1

    # Native runtime / hot path minimization
    native_performance_mode: bool = False
    native_batch_submit: bool = True
    native_max_batch_ops: int = 64
    native_flush_interval_ms: float = 0.25
    native_disable_python_callbacks: bool = False
    native_chunk_bytes: int = 64 * 1024 * 1024
    native_inflight_bytes_limit: int = 8 * 1024**3

    # Static spill plan / compile-capture
    static_plan_mode: bool = False
    static_plan_dir: str = ""
    static_plan_key: str = ""
    static_plan_auto_compile: bool = True
    compile_capture_parity_gate: bool = True
    graph_safe_mode: bool = True
    graph_ring_slots: int = 8
    graph_fixed_bucket_mb: int = 256

    # Distributed coordination
    distributed_coordination_enabled: bool = True
    coordination_dir: str = ""
    coordination_interval_steps: int = 5
    coordination_leader_rank: int = 0
    coordination_timeout_s: float = 1.0
    anti_skew_enabled: bool = True
    reproducibility_mode: bool = False
    topology_awareness_enabled: bool = True

    # Inference KV manager
    kv_manager_enabled: bool = False
    kv_budget_bytes: int = 8 * 1024**3
    kv_prefill_prefetch_lookahead: int = 8
    kv_decode_prefetch_lookahead: int = 1
    kv_latency_slo_ms: float = 20.0
    kv_eviction_policy: str = "LRU"  # LRU | CLOCK
    kv_tenant_fairness: bool = True
    kv_tenant_budget_ratio: float = 0.25

    # Control-plane / agent
    control_plane_dir: str = ""
    policy_name: str = ""
    policy_canary_ratio: float = 1.0
    agent_bind: str = "127.0.0.1"
    agent_port: int = 9765
