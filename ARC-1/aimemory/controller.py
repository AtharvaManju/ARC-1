import time
import threading
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
from .backend import detect_distributed, choose_backend
from .logging_jsonl import JsonlLogger, default_rank_log_path, safe_exc
from .gc import gc_windows_if_needed

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

        if cfg.require_license:
            st = verify_license(cfg.license_path)
            if not st.ok:
                raise RuntimeError(f"License check failed: {st.reason}")

        self.metrics = Metrics()
        self.metrics.baseline_step_ms_ema.alpha = float(cfg.baseline_ema_alpha)
        self.policy = AdaptivePolicy(self.metrics, cfg.overhead_sla_pct, cfg.safe_mode_cooldown_steps)
        self.pcc = DeterministicPCC(cfg.pcc_lookahead, cfg.pcc_drift_disable_threshold)

        self.backend = choose_backend(cfg.backend, cfg.pool_dir)
        self.metrics.backend = self.backend
        self._noop = (self.backend == "NOOP") or (not torch.cuda.is_available())

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
        else:
            self.storage = SARCStorage(
                pool_dir=cfg.pool_dir,
                rank=self.rank,
                staging_mb=cfg.staging_mb,
                durable=cfg.durable,
                encrypt_at_rest=cfg.encrypt_at_rest,
                key_path=cfg.encryption_key_path,
                strict_direct=cfg.strict_direct,
                backend=("RAM" if self.backend == "RAM" else "NVME_FILE"),
                pinned_pool=None,
                pool_window_steps=cfg.pool_window_steps,
                ram_max_bytes=cfg.ram_max_bytes,
            )
            self.io = IOWorkers(
                storage=self.storage,
                metrics=self.metrics,
                num_workers=cfg.io_workers,
                max_queue=cfg.max_queue,
                pinned_pool_bytes=cfg.pinned_pool_bytes,
                stream_priority=cfg.prefetch_stream_priority,
            )

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

        if self._logger:
            self._logger.log({"evt": "init", "rank": self.rank, "world": self.world_size, "backend": self.backend})

    def shutdown(self):
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
            "prefetch_hit_rate": m.prefetch_hit_rate(),
            "safe_mode": m.safe_mode,
            "disable_reason": m.disable_reason,
            "pcc_enabled": m.pcc_enabled,
            "pcc_drift_count": m.pcc_drift_count,
            "queue_depth": (self.io.queue_depth() if self.io else 0),
        }

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
        if not self.pcc.enabled or self.pcc.mode != "EXECUTION":
            return
        for pidx in self.pcc.next_prefetch_pack_indices():
            self._maybe_submit_prefetch_for_pack(int(pidx))

    def _pack_hook(self, t: torch.Tensor) -> PackedRef:
        self._pack_counter += 1
        pack_idx = self._pack_counter

        if self._noop or (not t.is_cuda):
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        nbytes = int(t.numel() * t.element_size())
        if (not self.metrics.spilling_enabled) or self.metrics.safe_mode or (nbytes < int(self.cfg.spill_min_bytes)):
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        if not t.is_contiguous():
            t = t.contiguous()

        key = self._new_key(pack_idx)
        self._step_keys_by_pack[pack_idx] = key

        dtype_s = dtype_to_str(t.dtype)
        shape = tuple(int(x) for x in t.shape)

        padded_plain = int(round_up(nbytes, 4096))

        assert self.io is not None
        blk = self.io.acquire_pinned(padded_plain)
        if blk is None:
            # Respect pinned pool budget: if no block is available, keep inline.
            self.metrics.spill_inline_pool_exhausted += 1
            return PackedRef(kind="INLINE", pack_idx=pack_idx, tensor=t)

        src_u8 = t.view(torch.uint8).reshape(-1)
        blk.u8[:nbytes].copy_(src_u8, non_blocking=True)

        ready_evt = torch.cuda.Event(enable_timing=False, blocking=False)
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

        self.metrics.spills += 1
        self.metrics.spill_bytes += nbytes
        return PackedRef(kind="SPILLED", pack_idx=pack_idx, key=key, dtype_s=dtype_s, shape=shape, nbytes=nbytes)

    def _unpack_hook(self, ref: PackedRef) -> torch.Tensor:
        if ref.kind == "INLINE":
            assert ref.tensor is not None
            return ref.tensor

        assert self.storage is not None and self.io is not None

        # FIX #6: fail fast if worker already hit fatal IO error
        fatal = self.io.fatal_error()
        if fatal is not None:
            raise RuntimeError(f"AIMemory IO fatal error (disable AIMemory or fix IO): {fatal}")

        if self._profiling:
            self.pcc.record_restore(ref.pack_idx)

        key = ref.key
        done_evt = self._spill_done_events.get(key)
        if done_evt is not None and not done_evt.is_set():
            t0 = time.time()
            done_evt.wait()
            dt_ms = (time.time() - t0) * 1000.0
            self.metrics.spill_commit_wait_ms_ema.update(dt_ms)

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
                meta = self.storage.get_meta(key)

                enc_blk = None
                scratch = None
                try:
                    enc_blk = self.storage.acquire_pinned(meta.padded_bytes)
                    self.storage.read_block_to_pinned(meta, enc_blk)

                    out = torch.empty(meta.shape, device="cuda", dtype=str_to_dtype(meta.dtype_s))
                    if meta.encrypted:
                        scratch = self.storage.acquire_pinned(meta.nbytes)

                    self.storage.restore_to_cuda_from_pinned(meta, enc_blk, out, scratch_plain=scratch)
                    out.record_stream(torch.cuda.current_stream())
                finally:
                    if enc_blk is not None:
                        self.storage.release_pinned(enc_blk)
                    if scratch is not None:
                        self.storage.release_pinned(scratch)

        except Exception as e:
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

            c.pcc.reset_step()

            if not c._noop:
                # CUDA event timing (no global synchronize)
                c._step_start_evt = torch.cuda.Event(enable_timing=True)
                c._step_end_evt = torch.cuda.Event(enable_timing=True)
                c._step_start_evt.record(torch.cuda.current_stream())

            if c._noop:
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
                        "backend": c.backend,
                        "rank": c.rank,
                    },
                    enabled=c.cfg.telemetry_opt_in,
                    offline_mode=c.cfg.offline_mode,
                    out_dir=c.cfg.telemetry_dir,
                )

                if c._logger:
                    c._logger.log({"evt": "step_end", "step": c._step_id, "dt_ms": dt_ms, "backend": c.backend})

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
