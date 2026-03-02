import tempfile

from aimemory.memory_trace import MemoryTraceRecorder


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_trace_") as td:
        rec = MemoryTraceRecorder(max_events=100, out_path=f"{td}/trace.json")
        rec.trace_pack(key="k1", nbytes=64 * 1024 * 1024, decision="spill", reason="queued_spill", step=1, pack_idx=1)
        rec.trace_pack(key="", nbytes=1024, decision="inline", reason="small_tensor_inline", step=1, pack_idx=2)
        rec.trace_restore(key="k1", nbytes=64 * 1024 * 1024, source="sync_restore", stall_ms=30.0, step=1, pack_idx=1)
        s = rec.summarize()
        assert s["events"] == 3
        assert "recommendations" in s
        p = rec.flush()
        assert p.endswith("trace.json")
    print("MEMORY_TRACE_OK")


if __name__ == "__main__":
    main()
