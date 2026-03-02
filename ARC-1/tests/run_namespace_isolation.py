import tempfile

import torch

from aimemory.storage import SARCStorage


def _write_one(st: SARCStorage, key: str):
    blk = st.acquire_pinned(4096)
    blk.u8[:4096].copy_(torch.arange(4096, dtype=torch.uint8), non_blocking=False)
    st.put_host_bytes(
        key=key,
        plain_blk=blk,
        nbytes=4096,
        dtype_s="uint8",
        shape=(4096,),
        step_id=0,
    )


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_ns_") as td:
        a = SARCStorage(pool_dir=td, rank=0, backend="NVME_FILE", tenant_namespace="tenant_a")
        b = SARCStorage(pool_dir=td, rank=0, backend="NVME_FILE", tenant_namespace="tenant_b")
        try:
            assert a.rank_dir != b.rank_dir
            _write_one(a, "k")
            _write_one(b, "k")
            ma = a.get_meta("k")
            mb = b.get_meta("k")
            assert ma.key == "k" and mb.key == "k"
            assert a.rank_dir.endswith("tenant_a/rank_0")
            assert b.rank_dir.endswith("tenant_b/rank_0")
        finally:
            a.close()
            b.close()
    print("NAMESPACE_ISOLATION_OK")


if __name__ == "__main__":
    main()
