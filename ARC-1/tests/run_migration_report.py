import os
import tempfile

from aimemory.migration import build_migration_report, write_migration_report


def main():
    with tempfile.TemporaryDirectory(prefix="arc1_migration_") as td:
        p = os.path.join(td, "x.py")
        with open(p, "w") as f:
            f.write("import aimemory\n")
        rep = build_migration_report(td, rewrite=False)
        assert rep["files_with_hits"] >= 1
        out = os.path.join(td, "rep.json")
        rep2 = write_migration_report(td, out_path=out, rewrite=True)
        assert os.path.exists(out)
        assert rep2["files_rewritten"] >= 1
    print("MIGRATION_REPORT_OK")


if __name__ == "__main__":
    main()
