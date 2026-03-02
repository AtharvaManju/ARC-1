import tempfile

from aimemory.consistency import run_consistency_check
from aimemory.storage import SARCStorage


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_cons_") as td:
        st = SARCStorage(pool_dir=td, rank=0, backend="NVME_FILE", durable=False, encrypt_at_rest=False)
        st.close()
        rep = run_consistency_check(pool_dir=td, rank=0, repair=False)
        assert rep["ok"] is True
    print("CONSISTENCY_CHECK_OK")


if __name__ == "__main__":
    main()
