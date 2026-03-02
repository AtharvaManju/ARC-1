import json
import os
import tempfile
import time
from typing import Any, Dict

import torch

from aimemory.backend import benchmark_path, detect_backend_capabilities
from aimemory.storage import SARCStorage


def run_fastpath_qualification(
    pool_dir: str,
    out_path: str,
    probe_mb: int = 64,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    caps = detect_backend_capabilities(pool_dir)
    probe = benchmark_path(pool_dir=pool_dir, probe_mb=int(probe_mb), probe_seconds=1.0)

    with tempfile.TemporaryDirectory(prefix="arc1_fastpath_", dir=pool_dir) as td:
        st = SARCStorage(
            pool_dir=td,
            rank=0,
            backend="NVME_FILE",
            durable=False,
            encrypt_at_rest=False,
            compression_codec="zlib",
            direct_restore_enabled=True,
        )
        try:
            nbytes = 2 * 1024 * 1024
            blk = st.acquire_pinned(nbytes)
            blk.u8[:nbytes].fill_(7)
            t0 = time.time()
            st.put_host_bytes(
                key="fastpath_probe",
                plain_blk=blk,
                nbytes=nbytes,
                dtype_s="uint8",
                shape=(nbytes,),
                step_id=0,
            )
            write_ms = (time.time() - t0) * 1000.0
            st.release_pinned(blk)
            meta = st.get_meta("fastpath_probe")

            direct_success = False
            direct_ms = 0.0
            fallback_ms = 0.0
            if torch.cuda.is_available():
                out = torch.empty(meta.shape, device="cuda", dtype=torch.uint8)
                t1 = time.time()
                direct_success = bool(st.try_restore_direct_to_cuda(meta, out))
                direct_ms = (time.time() - t1) * 1000.0
                if not direct_success:
                    enc = st.acquire_pinned(meta.padded_bytes)
                    t2 = time.time()
                    st.read_block_to_pinned(meta, enc)
                    st.restore_to_cuda_from_pinned(meta, enc, out)
                    torch.cuda.synchronize()
                    fallback_ms = (time.time() - t2) * 1000.0
                    st.release_pinned(enc)
            report = {
                "pool_dir": os.path.abspath(pool_dir),
                "backend_capabilities": caps,
                "backend_probe": probe,
                "engine_native": bool(st.engine._native is not None),
                "gds_enabled": bool(st.engine.gds_enabled()),
                "uring_enabled": bool(st.engine.uring_enabled()),
                "direct_restore_enabled": bool(st.direct_restore_enabled),
                "direct_restore_success": bool(direct_success),
                "last_decompress_path": str(st.last_decompress_path),
                "write_probe_ms": float(write_ms),
                "direct_restore_ms": float(direct_ms),
                "fallback_restore_ms": float(fallback_ms),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        finally:
            st.close()
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report
