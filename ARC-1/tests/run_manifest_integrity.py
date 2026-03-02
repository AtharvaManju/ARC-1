import tempfile
import torch

from aimemory.storage import SARCStorage


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_manifest_") as td:
        st = SARCStorage(
            pool_dir=td,
            rank=0,
            backend="NVME_FILE",
            durable=False,
            encrypt_at_rest=False,
            compression_codec="zlib",
            compression_min_bytes=1,
        )
        try:
            key = "manifest_k1"
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
            meta = st.get_meta(key)

            payload = st.acquire_pinned(meta.padded_bytes)
            st.read_block_to_pinned(meta, payload)
            payload.u8[0] = int(payload.u8[0].item() ^ 0x01)
            try:
                st.decode_to_plain_block(meta, payload)
                raise AssertionError("expected integrity verification failure")
            except RuntimeError as e:
                msg = str(e).lower()
                assert ("manifest" in msg) or ("crc32 mismatch" in msg), msg
            finally:
                st.release_pinned(payload)

            payload_ok = st.acquire_pinned(meta.padded_bytes)
            st.read_block_to_pinned(meta, payload_ok)
            plain_blk, work_blk, owns_plain, owns_work = st.decode_to_plain_block(meta, payload_ok)
            try:
                assert int(plain_blk.u8[10].item()) == 10
            finally:
                if owns_plain:
                    st.release_pinned(plain_blk)
                if owns_work and work_blk is not None:
                    st.release_pinned(work_blk)
                st.release_pinned(payload_ok)
        finally:
            st.close()
    print("MANIFEST_INTEGRITY_OK")


if __name__ == "__main__":
    main()
