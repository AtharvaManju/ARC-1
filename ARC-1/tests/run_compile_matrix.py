import os
import tempfile

from aimemory.bench.compile_matrix import run_compile_matrix


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_compile_matrix_") as td:
        out = os.path.join(td, "matrix.json")
        rep = run_compile_matrix(out_path=out, dims=[64], dtypes=["float32"], steps=2)
        assert "rows" in rep and len(rep["rows"]) == 1
        assert os.path.exists(out)
    print("COMPILE_MATRIX_OK")


if __name__ == "__main__":
    main()
