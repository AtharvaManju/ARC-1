import os
import tempfile
import threading
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch

from .config import AIMemoryConfig
from .controller import AIMemoryController

_LOCK = threading.Lock()
_TLS = threading.local()
_STATE: Dict[str, Any] = {
    "enabled": False,
    "controller": None,
    "pcc_finalized": False,
    "orig_tensor_backward": None,
    "orig_autograd_backward": None,
}


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


def _enter_step(profiling_warmup: bool):
    ctrl = _STATE.get("controller")
    if ctrl is None:
        return None
    in_step = bool(getattr(_TLS, "in_step", False))
    if in_step:
        return None
    _TLS.in_step = True
    ctx = ctrl.step(profiling_warmup=bool(profiling_warmup))
    ctx.__enter__()
    return ctx


def _exit_step(ctx, profiling_warmup: bool):
    if ctx is None:
        return
    try:
        ctx.__exit__(None, None, None)
    finally:
        _TLS.in_step = False
    if profiling_warmup and (not bool(_STATE.get("pcc_finalized", False))):
        ctrl = _STATE.get("controller")
        if ctrl is not None:
            try:
                ctrl.finalize_pcc_profile()
                _STATE["pcc_finalized"] = True
            except Exception:
                pass


def _patch_backprop():
    if _STATE.get("orig_tensor_backward") is None:
        _STATE["orig_tensor_backward"] = torch.Tensor.backward
    if _STATE.get("orig_autograd_backward") is None:
        _STATE["orig_autograd_backward"] = torch.autograd.backward

    def _tensor_backward_patched(tensor_self, *args, **kwargs):
        if not bool(_STATE.get("enabled", False)):
            return _STATE["orig_tensor_backward"](tensor_self, *args, **kwargs)
        profiling = not bool(_STATE.get("pcc_finalized", False))
        ctx = _enter_step(profiling_warmup=profiling)
        try:
            return _STATE["orig_tensor_backward"](tensor_self, *args, **kwargs)
        finally:
            _exit_step(ctx, profiling_warmup=profiling)

    def _autograd_backward_patched(*args, **kwargs):
        if not bool(_STATE.get("enabled", False)):
            return _STATE["orig_autograd_backward"](*args, **kwargs)
        profiling = not bool(_STATE.get("pcc_finalized", False))
        ctx = _enter_step(profiling_warmup=profiling)
        try:
            return _STATE["orig_autograd_backward"](*args, **kwargs)
        finally:
            _exit_step(ctx, profiling_warmup=profiling)

    torch.Tensor.backward = _tensor_backward_patched
    torch.autograd.backward = _autograd_backward_patched


def _unpatch_backprop():
    otb = _STATE.get("orig_tensor_backward")
    if otb is not None:
        torch.Tensor.backward = otb
    oab = _STATE.get("orig_autograd_backward")
    if oab is not None:
        torch.autograd.backward = oab


def enable(
    config: Optional[AIMemoryConfig] = None,
    **overrides: Any,
) -> AIMemoryController:
    """
    One-line integration:
      import arc1
      arc1.enable()
    """
    with _LOCK:
        if bool(_STATE.get("enabled", False)) and (_STATE.get("controller") is not None):
            return _STATE["controller"]

        if not _truthy_env("ARC1_ENABLE", True) or (not _truthy_env("AIMEMORY_ENABLE", True)):
            cfg = AIMemoryConfig(backend="NOOP")
        else:
            cfg = config if config is not None else AIMemoryConfig()
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            if os.environ.get("ARC1_POOL_DIR"):
                cfg.pool_dir = str(os.environ["ARC1_POOL_DIR"])
            if os.environ.get("ARC1_POLICY"):
                cfg.memory_slo_policy = str(os.environ["ARC1_POLICY"])
            # One-line mode should never fail on an unwritable default path.
            try:
                os.makedirs(cfg.pool_dir, exist_ok=True)
            except Exception:
                fallback = os.path.abspath(os.path.join(tempfile.gettempdir(), "arc1_pool"))
                os.makedirs(fallback, exist_ok=True)
                cfg.pool_dir = fallback

        ctrl = AIMemoryController(cfg)
        _STATE["controller"] = ctrl
        _STATE["enabled"] = True
        _STATE["pcc_finalized"] = False
        _patch_backprop()
        return ctrl


def disable(shutdown: bool = True):
    with _LOCK:
        _STATE["enabled"] = False
        _unpatch_backprop()
        ctrl = _STATE.get("controller")
        _STATE["controller"] = None
        _STATE["pcc_finalized"] = False
    if shutdown and ctrl is not None:
        try:
            ctrl.shutdown()
        except Exception:
            pass


def status() -> Dict[str, Any]:
    ctrl = _STATE.get("controller")
    snap = {}
    if ctrl is not None:
        try:
            snap = ctrl.metrics_snapshot()
        except Exception:
            snap = {}
    return {
        "enabled": bool(_STATE.get("enabled", False)),
        "pcc_finalized": bool(_STATE.get("pcc_finalized", False)),
        "has_controller": bool(ctrl is not None),
        "config": (asdict(ctrl.cfg) if ctrl is not None else {}),
        "metrics": snap,
    }
