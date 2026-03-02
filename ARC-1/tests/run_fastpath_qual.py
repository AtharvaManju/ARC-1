import os
import tempfile

from aimemory.bench.fastpath_qual import run_fastpath_qualification


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_fastpath_") as td:
        out = os.path.join(td, "fastpath.json")
        rep = run_fastpath_qualification(pool_dir=td, out_path=out, probe_mb=4)
        assert "backend_capabilities" in rep
        assert os.path.exists(out)
    print("FASTPATH_QUAL_OK")


if __name__ == "__main__":
    main()
