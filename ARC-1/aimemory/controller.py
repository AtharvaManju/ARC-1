import time
import threading
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .config import AIMemoryConfig
from .metrics import Metrics
from .policy import AdaptivePolicy
from .pcc import DeterministicPCC
from .storage import SARCStorage
from .io_workers import IOWorkers, SpillHostJob, PrefetchJob
from .util import dtype_to_str, str_to_dtype, round_up
from .license import verify_license
from .telemetry import emit_telemetry
from .backend import (
    detect_distributed,
    choose_backend,
    path_is_likely_remote,
    detect_backend_capabilities,
    benchmark_path,
    recommend_io_tuning,
)
from .logging_jsonl import JsonlLogger, default_rank_log_path, safe_exc
from .gc import gc_windows_if_needed
from .autotune import RuntimeAutoTuner
from .control_plane import PolicyStore, build_fleet_report
from .governor import MemoryGovernor
from .native_runtime import NativeRuntime
from .static_plan import StaticPlanCompiler, model_fingerprint_full
from .distributed_coord import RankCoordinator
from .kv_manager import KVResidencyManager
from .perf_envelope import PerformanceEnvelope
from .topology import detect_topology
from .allocator import allocator_snapshot
from .roi import ROITracker, WorkloadIdentity

@dataclass
class PackedRef:
    kind: str  # "INLINE" or "SPILLED"
    pack_idx: int
    tensor: Optional[torch.Tensor] = None
    key: str = ""
    dtype_s: str = ""
    shape: Tuple[int, ...] = ()
    nbytes: int = 0

class AIMemoryController:
    def __init__(self, cfg: AIMemoryConfig, rank: int = 0):
        self.cfg = cfg

        is_dist, det_rank, det_world, is_multinode = detect_distributed()
        self.rank = int(cfg.rank) if cfg.rank >= 0 else int(det_rank if is_dist else rank)
        self.world_size = int(cfg.world_size) if cfg.world_size >= 0 else int(det_world if is_dist else 1)

        if cfg.warn_on_multinode and is_multinode:
            print("[aimemory] WARNING: multi-node detected. Use per-node local pool_dir. Shared FS not supported.")
        if is_multinode and bool(getattr(cfg, "strict_local_pool", True)) and path_is_likely_remote(cfg.pool_dir):
            raise RuntimeError(f"pool_dir appears non-local for multi-node setup: {cfg.pool_dir}")

        cp_dir = str(getattr(cfg, "control_plane_dir", "") or os.path.join(cfg.pool_dir, "control_plane"))
        self._policy_store = PolicyStore(cp_dir)
        self._policy_name = str(getattr(cfg, "policy_name", ""))
        if self._policy_name:
            ratio = max(0.0, min(float(getattr(cfg, "policy_canary_ratio", 1.0)), 1.0))
            apply_pol = (ratio >= 1.0) or ((self.rank % 1000) < int(ratio * 1000))
            if apply_pol:
                pol = self._policy_store.select_for_rank(
                    name=self._policy_name,
                    rank=self.rank,
                    require_signature=bool(getattr(cfg, "policy_require_signature", False)),
                    key_uri=str(getattr(cfg, "policy_signing_key_uri", "")),
                    stage=str(getattr(cfg, "policy_stage", "")),
                )
                if pol:
                    self._policy_store.apply_to_config(cfg, pol)

        if cfg.require_license:
            st = verify_license(cfg.license_path)
            if not st.ok:
                raise RuntimeError(f"License check failed: {st.reason}")

        if bool(getattr(cfg, "reproducibility_mode", False)):
            cfg.deterministic_io_ordering = True
            cfg.deterministic_stream_fences = True

        self.metrics = Metrics()
        self.metrics.baseline_step_ms_ema.alpha = float(cfg.baseline_ema_alpha)
        self.policy = AdaptivePolicy(self.metrics, cfg.overhead_sla_pct, cfg.safe_mode_cooldown_steps)
        self.pcc = DeterministicPCC(cfg.pcc_lookahead, cfg.pcc_drift_disable_threshold)

        self.backend = choose_backend(
            cfg.backend,
            cfg.pool_dir,
            allow_tmpfs=bool(getattr(cfg, "allow_tmpfs_backend", True)),
            allow_network=bool(getattr(cfg, "allow_network_backend", False)),
            strict_local_pool=bool(getattr(cfg, "strict_local_pool", True)),
        )
        self._backend_caps = detect_backend_capabilities(cfg.pool_dir)
        self._backend_probe = benchmark_path(
            cfg.pool_dir,
            probe_mb=int(getattr(cfg, "backend_probe_mb", 128)),
            probe_seconds=float(getattr(cfg, "backend_probe_seconds", 2.0)),
        ) if self.backend in ("NVME_FILE", "TMPFS", "NETWORK_FILE") else {"ok": True}
        if bool(getattr(cfg, "backend_auto_disable_on_slow_fs", True)) and self.backend == "NETWORK_FILE":
            self.backend = "RAM"
        tune = recommend_io_tuning(self._backend_caps, self._backend_probe)
        if int(getattr(cfg, "max_queue", 0)) > int(tune.get("max_queue", cfg.max_queue)):
            cfg.max_queue = int(tune.get("max_queue", cfg.max_queue))
        if int(getattr(cfg, "io_workers", 0)) > int(tune.get("io_workers", cfg.io_workers)):
            cfg.io_workers = int(tune.get("io_workers", cfg.io_workers))
        cfg.native_chunk_bytes = int(tune.get("native_chunk_bytes", int(getattr(cfg, "native_chunk_bytes", 64 * 1024 * 1024))))
        self.metrics.backend = self.backend
        self._noop = (self.backend == "NOOP") or (not torch.cuda.is_available())
        env = self._policy_store.pull_envelope(self._policy_name) if self._policy_name else None
        if env is not None:
            self.metrics.policy_version = int(env.get("version", 0))

        self._logger = None
        if cfg.enable_jsonl_logs:
            try:
                self._logger = JsonlLogger(default_rank_log_path(cfg.pool_dir, self.rank))
            except Exception as e:
                print(f"[aimemory] logger disabled: {safe_exc(e)}")
                self._logger = None

        if self._noop:
            self.storage = None
            self.io = None
            self.kv = None
        else:
            self.storage = SARCStorage(
                pool_dir=cfg.pool_dir,
                rank=self.rank,
                staging_mb=cfg.staging_mb,
                durable=cfg.durable,
                encrypt_at_rest=cfg.encrypt_at_rest,
                key_path=cfg.encryption_key_path,
                key_uri=cfg.kms_key_uri,
                compression_codec=cfg.compression_codec,
                compression_min_bytes=cfg.compression_min_bytes,
                compression_min_gain_ratio=cfg.compression_min_gain_ratio,
                audit_log_path=cfg.audit_log_path,
                enable_audit_log=cfg.enable_audit_log,
                strict_direct=cfg.strict_direct,
                backend=("RAM" if self.backend == "RAM" else "NVME_FILE"),
                pinned_pool=None,
                pool_window_steps=cfg.pool_window_steps,
                ram_max_bytes=cfg.ram_max_bytes,
                tenant_namespace=str(getattr(cfg, "tenant_namespace", "default")),
                disk_quota_bytes=int(getattr(cfg, "disk_quota_bytes", cfg.max_pool_bytes)),
            )
            self.io = IOWorkers(
                storage=self.storage,
                metrics=self.metrics,
                num_workers=cfg.io_workers,
                max_queue=cfg.max_queue,
                pinned_pool_bytes=cfg.pinned_pool_bytes,
                stream_priority=cfg.prefetch_stream_priority,
                enforce_ordering=bool(getattr(cfg, "deterministic_io_ordering", True)),
                inflight_spill_bytes_limit=int(getattr(cfg, "inflight_spill_bytes_limit", 0)),
                inflight_prefetch_bytes_limit=int(getattr(cfg, "inflight_prefetch_bytes_limit", 0)),
                fairness_per_step_bytes=int(getattr(cfg, "fairness_per_step_bytes", 0)),
                nvme_write_bw_limit_mb_s=float(getattr(cfg, "nvme_write_bw_limit_mb_s", 0.0)),
                nvme_read_bw_limit_mb_s=float(getattr(cfg, "nvme_read_bw_limit_mb_s", 0.0)),
                queue_soft_limit=max(1, int(float(getattr(cfg, "queue_soft_limit_ratio", 0.85)) * max(1, int(cfg.max_queue)))),
            )
            self.kv = KVResidencyManager(cfg, self.storage)

        self._step_id = 0
        self._in_step = False
        self._profiling = False

        self._pack_counter = 0
        self._step_keys_by_pack: Dict[int, str] = {}
        self._prefetch_submitted_pack: Dict[int, bool] = {}
        self._spill_done_events: Dict[str, threading.Event] = {}

        self._hooks_ctx = None
        self._step_start_evt: Optional[torch.cuda.Event] = None
        self._step_end_evt: Optional[torch.cuda.Event] = None
        self._step_spills_start: int = 0
        ns = str(getattr(cfg, "tenant_namespace", "default")).strip().replace("/", "_").replace("..", "_")
        if ns and ns not in ("default", "shared"):
            self._agent_metrics_path = os.path.join(cfg.pool_dir, f"ns_{ns}", f"rank_{self.rank}", "agent_metrics.json")
        else:
            self._agent_metrics_path = os.path.join(cfg.pool_dir, f"rank_{self.rank}", "agent_metrics.json")
        self._roi = ROITracker(str(getattr(cfg, "roi_dir", "") or os.path.join(cfg.pool_dir, "roi")))
        self._wid = WorkloadIdentity(
            model=str(getattr(cfg, "model_profile_name", "") or "default"),
            batch_size=0,
            seq_len=0,
            precision="autograd",
            world_size=int(self.world_size),
            job=str(getattr(cfg, "workload_id", "") or ""),
        )
        self._autotuner = RuntimeAutoTuner(cfg, self.metrics)
        self._governor = MemoryGovernor(cfg, self.metrics)
        self._envelope = PerformanceEnvelope(cfg, self.metrics)
        self._native_runtime = None
        if (not self._noop) and bool(getattr(cfg, "native_performance_mode", False)):
            self._native_runtime = NativeRuntime(cfg=cfg, storage=self.storage, io=self.io, metrics=self.metrics)

        self._plan_compiler = StaticPlanCompiler(
            str(getattr(cfg, "static_plan_dir", "") or os.path.join(cfg.pool_dir, "static_plans"))
        )
        self._static_plan = None
        self._static_plan_entries: list[int] = []
        self._static_plan_lookup: dict[int, int] = {}
        self._static_plan_key = str(getattr(cfg, "static_plan_key", ""))
        if bool(getattr(cfg, "static_plan_mode", False)) and self._static_plan_key:
            self._static_plan = self._plan_compiler.load(self._static_plan_key)
            if self._static_plan is not None:
                self._static_plan_entries = [int(e.pack_idx) for e in self._static_plan.entries]
                self._static_plan_lookup = {int(e.pack_idx): int(e.prefetch_lookahead) for e in self._static_plan.entries}

        self._rank_coord = None
        self._topology = detect_topology(rank=self.rank)
        coord_dir = str(getattr(cfg, "coordination_dir", "") or os.path.join(cfg.pool_dir, "coordination"))
        if bool(getattr(cfg, "distributed_coordination_enabled", True)) and int(self.world_size) > 1:
            self._rank_coord = RankCoordinator(
                root_dir=coord_dir,
                rank=self.rank,
                world_size=self.world_size,
                leader_rank=int(getattr(cfg, "coordination_leader_rank", 0)),
            )
        self._spill_stream: Optional[torch.cuda.Stream] = None
        self._restore_stream: Optional[torch.cuda.Stream] = None
        self._spill_order_evt: Optional[torch.cuda.Event] = None
        if (not self._noop) and bool(getattr(cfg, "deterministic_stream_fences", True)):
            self._spill_stream = torch.cuda.Stream(priority=int(cfg.prefetch_stream_priority))
            self._restore_stream = torch.cuda.Stream(priority=int(cfg.prefetch_stream_priority))

        if self._logger:
            self._logger.log({"evt": "init", "rank": self.rank, "world": self.world_size, "backend": self.backend})

    def shutdown(self):
        try:
            if self._native_runtime is not None:
                self._native_runtime.close()
        except Exception:
            pass
        if self._noop:
            return
        try:
            assert self.io is not None
            self.io.shutdown()
        except Exception:
            pass
        try:
            if self.storage is not None:
                self.storage.close()
        except Exception:
            pass

    def finalize_pcc_profile(self):
        if self._noop:
            return
        self.pcc.finalize_profile()
        self.metrics.pcc_enabled = self.pcc.enabled
        if bool(getattr(self.cfg, "static_plan_mode", False)) and bool(getattr(self.cfg, "static_plan_auto_compile", True)):
            sched = (self.pcc.schedule.restore_pack_order if self.pcc.schedule is not None else [])
            if sched:
                mfp = model_fingerprint_full(
                    name=str(getattr(self.cfg, "model_profile_name", "") or "default"),
                    shape_sig=(len(sched),),
                    dtype_s="autograd",
                    world_size=int(self.world_size),
                    graph_key=str(getattr(self.cfg, "static_plan_key", "") or ""),
                )
                plan = self._plan_compiler.compile_from_restore_order(
                    model_fingerprint=mfp,
                    restore_order=[int(x) for x in sched],
                    lookahead=int(self.cfg.pcc_lookahead),
                )
                self._plan_compiler.save(plan)
                self._static_plan = plan
                self._static_plan_key = str(plan.plan_key)
                self._static_plan_entries = [int(e.pack_idx) for e in plan.entries]
                self._static_plan_lookup = {int(e.pack_idx): int(e.prefetch_lookahead) for e in plan.entries}

    def quick_summary(self) -> str:
        m = self.metrics
        return (
            f"backend={self.backend} step={self._step_id} spills={m.spills} restores={m.restores} "
            f"prefetch_hit_rate={m.prefetch_hit_rate():.1f}% safe_mode={m.safe_mode} "
            f"pcc={m.pcc_enabled} drift={m.pcc_drift_count} qdepth={(self.io.queue_depth() if self.io else 0)}"
        )

    def metrics_snapshot(self) -> Dict[str, Any]:
        m = self.metrics
        return {
            "backend": self.backend,
            "rank": self.rank,
            "world_size": self.world_size,
            "spills": m.spills,
            "restores": m.restores,
            "spill_bytes": m.spill_bytes,
            "restore_bytes": m.restore_bytes,
            "prefetch_submitted": m.prefetch_submitted,
            "prefetch_hits": m.prefetch_hits,
            "prefetch_misses": m.prefetch_misses,
            "prefetch_dropped": m.prefetch_dropped,
            "spill_queue_overflow": m.spill_queue_overflow,
            "spill_sync_fallback": m.spill_sync_fallback,
            "spill_inline_pool_exhausted": m.spill_inline_pool_exhausted,
            "autotune_updates": m.autotune_updates,
            "prefetch_hit_rate": m.prefetch_hit_rate(),
            "safe_mode": m.safe_mode,
            "disable_reason": m.disable_reason,
            "pcc_enabled": m.pcc_enabled,
            "pcc_drift_count": m.pcc_drift_count,
            "queue_depth": (self.io.queue_depth() if self.io else 0),
            "step_p95_ms": m.step_hist.pct(95),
            "step_p99_ms": m.step_hist.pct(99),
            "step_p999_ms": m.step_hist.pct(99.9),
            "spill_min_bytes": int(self.cfg.spill_min_bytes),
            "pcc_lookahead": int(self.cfg.pcc_lookahead),
            "governor_level": int(m.governor_level),
            "governor_adjustments": int(m.governor_adjustments),
            "oom_degrade_count": int(m.oom_degrade_count),
            "memory_headroom_pct": float(m.memory_headroom_pct),
            "memory_free_bytes": int(m.memory_free_bytes),
            "memory_total_bytes": int(m.memory_total_bytes),
            "allocator_allocated_bytes": int(m.allocator_allocated_bytes),
            "allocator_reserved_bytes": int(m.allocator_reserved_bytes),
            "allocator_inactive_split_bytes": int(m.allocator_inactive_split_bytes),
            "allocator_reclaimable_bytes": int(m.allocator_reclaimable_bytes),
            "allocator_fragmentation_pct": float(m.allocator_fragmentation_pct),
            "static_plan_hits": int(m.static_plan_hits),
            "coord_policy_applied": int(m.coord_policy_applied),
            "kv_spills": int(m.kv_spills),
            "kv_restores": int(m.kv_restores),
            "throttle_events": int(m.throttle_events),
            "spill_budget_denials": int(m.spill_budget_denials),
            "inflight_budget_denials": int(m.inflight_budget_denials),
            "fairness_denials": int(m.fairness_denials),
            "quarantine_events": int(m.quarantine_events),
            "policy_version": int(m.policy_version),
            "native_runtime": (self._native_runtime.stats() if self._native_runtime is not None else {"enabled": False}),
            "static_plan_mode": bool(getattr(self.cfg, "static_plan_mode", False)),
            "static_plan_key": str(self._static_plan.plan_key) if self._static_plan is not None else "",
            "distributed_coordination": bool(self._rank_coord is not None),
            "reproducibility_mode": bool(getattr(self.cfg, "reproducibility_mode", False)),
            "collective_safe_mode": bool(getattr(self.cfg, "collective_safe_mode", True)),
            "topology": dict(self._topology),
            "kv_manager": (self.kv.stats() if getattr(self, "kv", None) is not None else {"enabled": False}),
            "latency_attribution": m.latency_snapshot(),
            "performance_envelope": self._envelope.snapshot(),
            "backpressure": (self.io.backpressure_state() if self.io is not None else {}),
            "engine_native": bool((self.storage is not None) and (self.storage.engine._native is not None)),
            "engine_gds": bool((self.storage is not None) and self.storage.engine.gds_enabled()),
            "engine_uring": bool((self.storage is not None) and self.storage.engine.uring_enabled()),
            "backend_capabilities": dict(self._backend_caps),
            "backend_probe": dict(self._backend_probe),
            "roi": self._roi.attribution(
                self._wid,
                {
                    "memory_headroom_pct": float(m.memory_headroom_pct),
                    "oom_events": int(m.oom_degrade_count),
                    "reruns": int(1 if m.safe_mode else 0),
                    "throughput": float(1000.0 / max(1e-6, m.step_ms_ema.value)) if m.step_ms_ema.value > 0 else 0.0,
                    "step_samples_ms": list(m.step_hist.xs),
                },
            ),
        }

    def compile_capture_parity_gate(self, baseline: Dict[str, float], candidate: Dict[str, float], tol_ratio: float = 0.05) -> Dict[str, Any]:
        return self._plan_compiler.compile_capture_parity(baseline=baseline, candidate=candidate, tol_ratio=tol_ratio)

    def kv_register(self, key: str, tensor: torch.Tensor):
        if getattr(self, "kv", None) is not None:
            self.kv.register(key, tensor)

    def kv_get(self, key: str) -> Optional[torch.Tensor]:
        if getattr(self, "kv", None) is not None:
            return self.kv.get(key)
        return None

    def kv_release(self, key: str):
        if getattr(self, "kv", None) is not None:
            self.kv.release(key)

    def _is_compiling(self) -> bool:
        try:
            import torch._dynamo as dynamo  # type: ignore
            return bool(dynamo.is_compiling())
        except Exception:
            return False

    def _new_key(self, pack_idx: int) -> str:
        return f"r{self.rank}_s{self._step_id}_p{pack_idx}_{int(time.time()*1e6)}"

    def _maybe_submit_prefetch_for_pack(self, pack_idx: int):
        assert self.storage is not None and self.io is not None
        if pack_idx in self._prefetch_submitted_pack:
            return
        if pack_idx not in self._step_keys_by_pack:
            return
        key = self._step_keys_by_pack[pack_idx]

        done_evt = self._spill_done_events.get(key)
        if done_evt is not None and not done_evt.is_set():
            return

        try:
            meta = self.storage.get_meta(key)
        except Exception:
            return

        if not self.io.reserve_prefetch(key):
            self._prefetch_submitted_pack[pack_idx] = True
            return

        self.metrics.prefetch_submitted += 1
        self._prefetch_submitted_pack[pack_idx] = True
        if not self.io.submit_prefetch(PrefetchJob(key=key, meta=meta)):
            self.metrics.prefetch_dropped += 1

    def _submit_lookahead_prefetch(self):
        if bool(getattr(self.cfg, "static_plan_mode", False)) and self._static_plan is not None:
            lim = max(1, int(getattr(self.cfg, "prefetch_batch_limit", 8)))
            for pidx in self._static_plan_entries[:lim]:
                self._maybe_submit_prefetch_for_pack(int(pidx))
                self.metrics.static_plan_hits += 1
            return
        if not self.pcc.enabled or self.pcc.mode != "EXECUTION":
            return
        if int(getattr(self.metrics, "governor_level", 0)) >= 2:
            return
        lim = max(1, int(getattr(self.cfg, "prefetch_batch_limit", 8)))
        if bool(getattr(self.cfg, "collective_safe_mode", True)):
            cad = max(1, int(getattr(self.cfg, "collective_cadence_steps", 1)))
            if (self._step_id % cad) == 0:
                lim = 1
        if self.io is not None:
            qd = int(self.io.queue_depth())
            q_cap = max(1, int(getattr(self.cfg, "max_queue", 1)))
            if qd >= int(0.75 * q_cap):
                lim = 1
        for pidx in self.pcc.next_prefetch_pack_indices()[:lim]:
            self._maybe_submit_prefetch_for_pack(int(pidx))

    def _pack_hook(self, t: torch.Tensor) -> PackedRef:
        self._pack_counter += 1
        pack_idx = self._pack_counter

        if bool(getattr(self.cfg, "compile_safe_mode", True)) and self._is_compiling():
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        if self._noop or (not t.is_cuda):
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        nbytes = int(t.numel() * t.element_size())
        if (not self.metrics.spilling_enabled) or self.metrics.safe_mode or (nbytes < int(self.cfg.spill_min_bytes)):
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)
        ok_budget, why = self._envelope.can_spill(nbytes=nbytes, rank=self.rank)
        if not ok_budget:
            self.metrics.disable_reason = f"spill_denied:{why}"
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)
        if self.io is not None:
            fatal = self.io.fatal_error()
            if fatal is not None and bool(getattr(self.cfg, "fail_open_on_error", True)):
                self.metrics.spilling_enabled = False
                self.metrics.safe_mode = True
                self.metrics.disable_reason = f"Fail-open after IO error: {fatal}"
                return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)
        if self._native_runtime is not None:
            nf = self._native_runtime.fatal()
            if nf is not None and bool(getattr(self.cfg, "fail_open_on_error", True)):
                self.metrics.spilling_enabled = False
                self.metrics.safe_mode = True
                self.metrics.disable_reason = f"Fail-open after native runtime error: {nf}"
                return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        if not t.is_contiguous():
            t = t.contiguous()

        key = self._new_key(pack_idx)
        self._step_keys_by_pack[pack_idx] = key

        dtype_s = dtype_to_str(t.dtype)
        shape = tuple(int(x) for x in t.shape)

        if self._native_runtime is not None and bool(getattr(self.cfg, "native_performance_mode", False)):
            done_evt = self.io.spill_done_event(key)
            self._spill_done_events[key] = done_evt
            submitted = self._native_runtime.submit_spill(key=key, step_id=self._step_id, t=t, done_evt=done_evt)
            if submitted:
                self._envelope.note_spill(nbytes=nbytes, rank=self.rank)
                self.metrics.spills += 1
                self.metrics.spill_bytes += nbytes
                return PackedRef(kind="SPILLED", pack_idx=pack_idx, key=key, dtype_s=dtype_s, shape=shape, nbytes=nbytes)

        padded_plain = int(round_up(nbytes, 4096))

        assert self.io is not None
        blk = self.io.acquire_pinned(padded_plain)
        if blk is None:
            # Respect pinned pool budget: if no block is available, keep inline.
            self.metrics.spill_inline_pool_exhausted += 1
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        src_u8 = t.view(torch.uint8).reshape(-1)
        ready_evt = torch.cuda.Event(enable_timing=False, blocking=False)
        if self._spill_stream is not None:
            gate_evt = torch.cuda.Event(enable_timing=False, blocking=False)
            gate_evt.record(torch.cuda.current_stream())
            with torch.cuda.stream(self._spill_stream):
                self._spill_stream.wait_event(gate_evt)
                if self._spill_order_evt is not None:
                    self._spill_stream.wait_event(self._spill_order_evt)
                blk.u8[:nbytes].copy_(src_u8, non_blocking=True)
                ready_evt.record(self._spill_stream)
            self._spill_order_evt = ready_evt
        else:
            blk.u8[:nbytes].copy_(src_u8, non_blocking=True)
            ready_evt.record(torch.cuda.current_stream())

        done_evt = self.io.spill_done_event(key)
        self._spill_done_events[key] = done_evt

        policy = str(getattr(self.cfg, "spill_queue_overflow_policy", "SYNC_SPILL")).upper()
        timeout_s = None if policy == "BLOCK" else float(getattr(self.cfg, "queue_put_timeout_s", 0.01))
        queued = self.io.submit_spill_host(
            SpillHostJob(
                key=key,
                step_id=self._step_id,
                pinned_block=blk,
                nbytes=nbytes,
                dtype_s=dtype_s,
                shape=shape,
                ready_evt=ready_evt,
                done_evt=done_evt,
            ),
            timeout_s=timeout_s,
        )

        if not queued:
            self.metrics.spill_queue_overflow += 1
            if policy == "DISABLE_SPILLING":
                ready_evt.synchronize()
                self.io.release_pinned(blk)
                done_evt.set()
                self.io.clear_spill_done(key)
                self._spill_done_events.pop(key, None)
                self._step_keys_by_pack.pop(pack_idx, None)
                self.metrics.spilling_enabled = False
                self.metrics.safe_mode = True
                self.metrics.disable_reason = "Queue overflow: spilling disabled"
                return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

            # SYNC_SPILL fallback: write spill in caller thread to preserve correctness.
            if policy == "SYNC_SPILL":
                assert self.storage is not None
                ready_evt.synchronize()
                try:
                    self.storage.put_host_bytes(
                        key=key,
                        plain_blk=blk,
                        nbytes=nbytes,
                        dtype_s=dtype_s,
                        shape=shape,
                        step_id=self._step_id,
                        spill_done_evt=done_evt,
                    )
                    self.metrics.spill_sync_fallback += 1
                finally:
                    self.io.release_pinned(blk)
                    self.io.clear_spill_done(key)
            else:
                # Unknown policy: fail safe to inline.
                ready_evt.synchronize()
                self.io.release_pinned(blk)
                done_evt.set()
                self.io.clear_spill_done(key)
                self._spill_done_events.pop(key, None)
                self._step_keys_by_pack.pop(pack_idx, None)
                return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        if bool(getattr(self.cfg, "pack_wait_for_commit", False)):
            done_evt.wait()
            fatal = self.io.fatal_error()
            if fatal is not None:
                if bool(getattr(self.cfg, "fail_open_on_error", True)):
                    self.metrics.spilling_enabled = False
                    self.metrics.safe_mode = True
                    self.metrics.disable_reason = f"Fail-open after commit error: {fatal}"
                    self.io.clear_spill_done(key)
                    self._spill_done_events.pop(key, None)
                    self._step_keys_by_pack.pop(pack_idx, None)
                    return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)
                raise RuntimeError(f"AIMemory spill commit failed for key={key}: {fatal}")

        self.metrics.spills += 1
        self.metrics.spill_bytes += nbytes
        self._envelope.note_spill(nbytes=nbytes, rank=self.rank)
        return PackedRef(kind="SPILLED", pack_idx=pack_idx, key=key, dtype_s=dtype_s, shape=shape, nbytes=nbytes)

    def _unpack_hook(self, ref: PackedRef) -> torch.Tensor:
        if ref.kind == "INLINE":
            assert ref.tensor is not None
            return ref.tensor

        assert self.storage is not None and self.io is not None

        fatal = self.io.fatal_error()
        if fatal is not None and bool(getattr(self.cfg, "fail_open_on_error", True)):
            self.metrics.spilling_enabled = False
            self.metrics.safe_mode = True
            self.metrics.disable_reason = f"Fail-open after IO error: {fatal}"
        if self._native_runtime is not None:
            nf = self._native_runtime.fatal()
            if nf is not None and bool(getattr(self.cfg, "fail_open_on_error", True)):
                self.metrics.spilling_enabled = False
                self.metrics.safe_mode = True
                self.metrics.disable_reason = f"Fail-open after native runtime error: {nf}"

        if self._profiling:
            self.pcc.record_restore(ref.pack_idx)

        key = ref.key
        done_evt = self._spill_done_events.get(key)
        if done_evt is not None and not done_evt.is_set():
            t0 = time.time()
            done_evt.wait()
            dt_ms = (time.time() - t0) * 1000.0
            self.metrics.spill_commit_wait_ms_ema.update(dt_ms)
            self.metrics.restore_wait_hist.add(dt_ms)

        out = None
        try:
            out = self.io.take_prefetched(key, timeout_s=0.0)
        except Exception:
            out = None

        try:
            if out is not None:
                self.metrics.prefetch_hits += 1
            else:
                self.metrics.prefetch_misses += 1
                try:
                    meta = self.storage.get_meta(key)
                except KeyError as e:
                    if not self.storage.wait_until_readable(
                        key=key,
                        timeout_s=float(getattr(self.cfg, "unpack_meta_wait_timeout_s", 2.0)),
                        poll_s=float(getattr(self.cfg, "unpack_meta_wait_poll_s", 0.002)),
                    ):
                        fatal = self.io.fatal_error()
                        if fatal is not None:
                            raise RuntimeError(f"AIMemory spill commit failed for key={key}: {fatal}") from e
                        raise
                    meta = self.storage.get_meta(key)

                enc_blk = None
                plain_blk = None
                work_blk = None
                owns_plain = False
                owns_work = False
                try:
                    enc_blk = self.storage.acquire_pinned(meta.padded_bytes)
                    try:
                        self.storage.mark_prefetching(key)
                    except Exception:
                        pass
                    t0_io = time.time()
                    self.storage.read_block_to_pinned(meta, enc_blk)
                    self.metrics.io_read_hist.add((time.time() - t0_io) * 1000.0)

                    t0_decode = time.time()
                    plain_blk, work_blk, owns_plain, owns_work = self.storage.decode_to_plain_block(
                        meta, enc_blk, scratch_plain=None, scratch_work=None
                    )
                    dt_decode = (time.time() - t0_decode) * 1000.0
                    self.metrics.decode_hist.add(dt_decode)
                    if int(getattr(meta, "compressed", 0)) == 1:
                        self.metrics.decompress_hist.add(dt_decode)

                    out = torch.empty(meta.shape, device="cuda", dtype=str_to_dtype(meta.dtype_s))
                    src_u8 = plain_blk.u8[:meta.nbytes].view(torch.uint8).reshape(-1)
                    out_u8 = out.view(torch.uint8).reshape(-1)
                    t0_h2d = time.time()
                    if self._restore_stream is not None:
                        evt = torch.cuda.Event(enable_timing=False, blocking=False)
                        with torch.cuda.stream(self._restore_stream):
                            out_u8.copy_(src_u8, non_blocking=True)
                            evt.record(self._restore_stream)
                        t0_sync = time.time()
                        torch.cuda.current_stream().wait_event(evt)
                        self.metrics.stream_sync_hist.add((time.time() - t0_sync) * 1000.0)
                    else:
                        out_u8.copy_(src_u8, non_blocking=True)
                    self.metrics.h2d_copy_hist.add((time.time() - t0_h2d) * 1000.0)
                    try:
                        self.storage.mark_resident(key)
                    except Exception:
                        pass
                    self.storage.mark_restored(key)
                    out.record_stream(torch.cuda.current_stream())
                finally:
                    if enc_blk is not None:
                        self.storage.release_pinned(enc_blk)
                    if owns_plain and plain_blk is not None:
                        self.storage.release_pinned(plain_blk)
                    if owns_work and work_blk is not None:
                        self.storage.release_pinned(work_blk)

        except Exception as e:
            if "out of memory" in str(e).lower():
                self._governor.on_oom_signal(self._step_id)
            if ("mismatch" in str(e).lower()) or ("corruption" in str(e).lower()):
                self.metrics.quarantine_events += 1
            self.metrics.spilling_enabled = False
            self.metrics.safe_mode = True
            self.metrics.disable_reason = f"Restore failed: {type(e).__name__}: {e}"
            if self._logger:
                self._logger.log({"evt": "restore_fail", "err": safe_exc(e), "key": key})
            if self.cfg.hard_fail_on_corruption:
                raise
            raise RuntimeError(f"AIMemory restore failed: {type(e).__name__}: {e}") from e

        self.metrics.restores += 1
        self.metrics.restore_bytes += int(ref.nbytes)

        if (not self._profiling) and self.pcc.mode == "EXECUTION" and self.pcc.enabled:
            self.pcc.advance_on_restore(ref.pack_idx)
            self.metrics.pcc_drift_count = self.pcc.drift_count
            self.metrics.pcc_enabled = self.pcc.enabled
            self._submit_lookahead_prefetch()

        return out

    class _StepCtx:
        def __init__(self, ctrl: "AIMemoryController", profiling_warmup: bool):
            self.ctrl = ctrl
            self.profiling_warmup = bool(profiling_warmup)

        def __enter__(self):
            c = self.ctrl
            if c._in_step:
                raise RuntimeError("Nested ctrl.step() not supported")
            c._in_step = True
            c._profiling = self.profiling_warmup

            c._pack_counter = 0
            c._step_keys_by_pack.clear()
            c._prefetch_submitted_pack.clear()
            c._spill_done_events.clear()
            c._step_spills_start = int(c.metrics.spills)
            c._spill_order_evt = None

            c.pcc.reset_step()
            c._envelope.start_step(c._step_id)

            if not c._noop:
                # CUDA event timing (no global synchronize)
                c._step_start_evt = torch.cuda.Event(enable_timing=True)
                c._step_end_evt = torch.cuda.Event(enable_timing=True)
                c._step_start_evt.record(torch.cuda.current_stream())

            if c._noop:
                return self

            if bool(getattr(c.cfg, "native_disable_python_callbacks", False)):
                # Explicit bypass mode for environments that run a native/autograph integration.
                c.metrics.disable_reason = "python_callbacks_bypassed"
                return self

            c._hooks_ctx = torch.autograd.graph.saved_tensors_hooks(c._pack_hook, c._unpack_hook)
            c._hooks_ctx.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            c = self.ctrl
            try:
                if c._hooks_ctx is not None:
                    c._hooks_ctx.__exit__(exc_type, exc, tb)
            finally:
                c._hooks_ctx = None

            if not c._noop:
                if c.cfg.sync_each_step:
                    # Optional full device sync (debug/bench)
                    assert c._step_end_evt is not None and c._step_start_evt is not None
                    c._step_end_evt.record(torch.cuda.current_stream())
                    torch.cuda.synchronize()
                    dt_ms = float(c._step_start_evt.elapsed_time(c._step_end_evt))
                else:
                    # Stream-local timing: record end event and wait only for this stream
                    assert c._step_end_evt is not None and c._step_start_evt is not None
                    c._step_end_evt.record(torch.cuda.current_stream())
                    c._step_end_evt.synchronize()
                    dt_ms = float(c._step_start_evt.elapsed_time(c._step_end_evt))

                step_used_spill = (int(c.metrics.spills) > int(c._step_spills_start))
                c.policy.update_step_baseline(dt_ms, with_aimemory=step_used_spill)
                c.metrics.step_hist.add(dt_ms)
                c.metrics.step_ms_ema.update(dt_ms)
                c.metrics.io_queue_depth_ema.update(float(c.io.queue_depth() if c.io else 0))

                c.policy.enforce_sla(c._step_id, dt_ms)
                c.policy.maybe_reenable(c._step_id)
                c._autotuner.observe_step(c._step_id)
                c._governor.observe_step(c._step_id)
                alloc = allocator_snapshot()
                c.metrics.allocator_allocated_bytes = int(alloc.get("allocated_bytes", 0))
                c.metrics.allocator_reserved_bytes = int(alloc.get("reserved_bytes", 0))
                c.metrics.allocator_inactive_split_bytes = int(alloc.get("inactive_split_bytes", 0))
                c.metrics.allocator_reclaimable_bytes = int(alloc.get("reclaimable_bytes", 0))
                c.metrics.allocator_fragmentation_pct = float(alloc.get("fragmentation_pct", 0.0))
                if bool(getattr(c.cfg, "allocator_friendly_mode", True)):
                    frag_warn = float(getattr(c.cfg, "allocator_fragmentation_warn_pct", 35.0))
                    if c.metrics.allocator_fragmentation_pct >= frag_warn:
                        c.cfg.pcc_lookahead = max(1, int(c.cfg.pcc_lookahead) - 1)
                        c.cfg.spill_min_bytes = max(1 * 1024 * 1024, int(c.cfg.spill_min_bytes * 0.9))
                c._envelope.end_step(io=c.io)
                c.pcc.lookahead = int(c.cfg.pcc_lookahead)
                if c._rank_coord is not None and (c._step_id % max(1, int(getattr(c.cfg, "coordination_interval_steps", 5))) == 0):
                    try:
                        c._rank_coord.publish(c._step_id, c.metrics_snapshot(), topology=(c._topology if bool(getattr(c.cfg, "topology_awareness_enabled", True)) else {}))
                        if c.rank == int(getattr(c.cfg, "coordination_leader_rank", 0)):
                            c._rank_coord.leader_aggregate(c._step_id)
                        consensus = c._rank_coord.poll_consensus(max_age_s=float(getattr(c.cfg, "coordination_timeout_s", 1.0)))
                        if consensus is not None:
                            applied = c._rank_coord.apply(
                                c.cfg,
                                consensus,
                                anti_skew=bool(getattr(c.cfg, "anti_skew_enabled", True)),
                                rank=c.rank,
                            )
                            if applied:
                                c.metrics.coord_policy_applied += 1
                    except Exception as e:
                        if c._logger:
                            c._logger.log({"evt": "coord_warn", "err": safe_exc(e)})
                if getattr(c, "kv", None) is not None:
                    try:
                        kst = c.kv.stats()
                        c.metrics.kv_spills = int(kst.get("spills", 0))
                        c.metrics.kv_restores = int(kst.get("restores", 0))
                    except Exception:
                        pass

                emit_telemetry(
                    {
                        "step": c._step_id,
                        "dt_ms": dt_ms,
                        "spills": c.metrics.spills,
                        "restores": c.metrics.restores,
                        "prefetch_hit_rate": c.metrics.prefetch_hit_rate(),
                        "safe_mode": c.metrics.safe_mode,
                        "disable_reason": c.metrics.disable_reason,
                        "pcc_enabled": c.metrics.pcc_enabled,
                        "pcc_drift": c.metrics.pcc_drift_count,
                        "governor_level": c.metrics.governor_level,
                        "memory_headroom_pct": c.metrics.memory_headroom_pct,
                        "step_p95_ms": c.metrics.step_hist.pct(95),
                        "step_p99_ms": c.metrics.step_hist.pct(99),
                        "step_p999_ms": c.metrics.step_hist.pct(99.9),
                        "restore_wait_p99_ms": c.metrics.restore_wait_hist.pct(99),
                        "io_write_p99_ms": c.metrics.io_write_hist.pct(99),
                        "io_read_p99_ms": c.metrics.io_read_hist.pct(99),
                        "decode_p99_ms": c.metrics.decode_hist.pct(99),
                        "allocator_fragmentation_pct": c.metrics.allocator_fragmentation_pct,
                        "policy_version": c.metrics.policy_version,
                        "backend": c.backend,
                        "rank": c.rank,
                    },
                    enabled=c.cfg.telemetry_opt_in,
                    offline_mode=c.cfg.offline_mode,
                    out_dir=c.cfg.telemetry_dir,
                )

                if c._logger:
                    c._logger.log({"evt": "step_end", "step": c._step_id, "dt_ms": dt_ms, "backend": c.backend})
                try:
                    os.makedirs(os.path.dirname(c._agent_metrics_path) or ".", exist_ok=True)
                    snap = c.metrics_snapshot()
                    snap["ts"] = float(time.time())
                    with open(c._agent_metrics_path, "w") as f:
                        json.dump(snap, f, indent=2)
                except Exception:
                    pass
                if (
                    c.rank == 0
                    and bool(getattr(c.cfg, "policy_auto_rollback_enabled", True))
                    and bool(c._policy_name)
                    and (c._step_id % max(5, int(getattr(c.cfg, "coordination_interval_steps", 5))) == 0)
                ):
                    try:
                        fleet = build_fleet_report(c.cfg.pool_dir)
                        rolled = c._policy_store.auto_rollback_on_slo(
                            c._policy_name,
                            fleet,
                            p99_ms_max=float(getattr(c.cfg, "policy_rollback_p99_ms", 0.0)),
                            safe_mode_max=int(getattr(c.cfg, "policy_rollback_safe_mode_max", 0)),
                        )
                        if rolled is not None and c._logger:
                            c._logger.log({"evt": "policy_auto_rollback", "name": c._policy_name})
                    except Exception as e:
                        if c._logger:
                            c._logger.log({"evt": "policy_auto_rollback_warn", "err": safe_exc(e)})

                # Deterministic retention GC (periodic + emergency budget)
                try:
                    gc_windows_if_needed(
                        pool_dir=c.cfg.pool_dir,
                        rank=c.rank,
                        current_step=c._step_id,
                        window_steps=c.cfg.pool_window_steps,
                        keep_last_windows=c.cfg.gc_keep_last_windows,
                        max_pool_bytes=c.cfg.max_pool_bytes,
                        every_steps=c.cfg.gc_every_steps,
                        checkpoint_truncate=c.cfg.gc_checkpoint_truncate,
                        vacuum=c.cfg.gc_vacuum,
                    )
                    if c.storage is not None and c.backend == "RAM":
                        cur_pool_id = c.storage._pool_id_for_step(c._step_id)
                        c.storage.ram_gc_keep_last_windows(c.cfg.gc_keep_last_windows, cur_pool_id)
                except Exception as e:
                    if c._logger:
                        c._logger.log({"evt": "gc_warn", "err": safe_exc(e)})

            c._step_id += 1
            c._in_step = False
            c._profiling = False
            return False

    def step(self, profiling_warmup: bool = False):
        return AIMemoryController._StepCtx(self, profiling_warmup=profiling_warmup)
