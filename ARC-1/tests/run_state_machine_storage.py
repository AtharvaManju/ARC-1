import tempfile
import torch

from aimemory.storage import SARCStorage, STATUS_EVICTABLE, STATUS_RESTORED


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_state_") as td:
        st = SARCStorage(
            pool_dir=td,
            rank=0,
            backend="NVME_FILE",
            durable=False,
            encrypt_at_rest=False,
            compression_codec="zlib",
            compression_min_bytes=1,
        )
        blk = st.acquire_pinned(1024)
        blk.u8[:1024].copy_(torch.arange(1024, dtype=torch.uint8), non_blocking=False)
        meta = st.put_host_bytes(
            key="k1",
            plain_blk=blk,
            nbytes=1024,
            dtype_s="uint8",
            shape=(1024,),
            step_id=0,
        )
        got = st.get_meta("k1")
        assert got.status_s == STATUS_EVICTABLE
        assert int(got.stored_nbytes) > 0
        st.mark_restored("k1")
        got2 = st.get_meta("k1")
        assert got2.status_s == STATUS_RESTORED
        rep = st.consistency_report(repair=False)
        assert rep["ok"] is True
        st.close()
    print("STATE_MACHINE_STORAGE_OK")


if __name__ == "__main__":
    main()
