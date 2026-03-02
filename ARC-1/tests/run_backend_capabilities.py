import tempfile

from aimemory.backend import detect_backend_capabilities, benchmark_path, recommend_io_tuning


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_backend_") as td:
        caps = detect_backend_capabilities(td)
        assert caps["writable"] is True
        probe = benchmark_path(td, probe_mb=4, probe_seconds=0.25)
        rec = recommend_io_tuning(caps, probe)
        assert "backend" in rec and "max_queue" in rec
    print("BACKEND_CAPABILITIES_OK")


if __name__ == "__main__":
    main()
