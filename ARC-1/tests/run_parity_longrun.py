import os
import tempfile

from aimemory.bench.parity_longrun import run_parity_longrun


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_parity_longrun_") as td:
        out = os.path.join(td, "parity.json")
        rep = run_parity_longrun(pool_dir=td, out_path=out, steps=4, dim=64, dtype_s="float32")
        assert "parity" in rep
        assert os.path.exists(out)
    print("PARITY_LONGRUN_OK")


if __name__ == "__main__":
    main()
