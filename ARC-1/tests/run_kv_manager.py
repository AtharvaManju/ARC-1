import tempfile

import torch

from aimemory.config import AIMemoryConfig
from aimemory.kv_manager import KVResidencyManager
from aimemory.storage import SARCStorage


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_kv_") as td:
        st = SARCStorage(pool_dir=td, rank=0, backend="NVME_FILE", encrypt_at_rest=False, durable=False, compression_codec="zlib", compression_min_bytes=1)
        cfg = AIMemoryConfig(kv_manager_enabled=True, kv_budget_bytes=4096)
        kv = KVResidencyManager(cfg, st)
        kv.set_phase("prefill")

        for i in range(4):
            t = torch.arange(2048, dtype=torch.uint8).clone()
            kv.register(f"k{i}", t)
        s = kv.stats()
        assert s["spilled_blocks"] >= 1

        # Access should restore spilled block if needed.
        got = kv.get("k0")
        assert got is not None
        assert int(got.view(torch.uint8)[10].item()) == 10

        kv.set_phase("decode")
        _ = kv.get("k0")
        s2 = kv.stats()
        assert s2["phase"] == "decode"
        st.close()
    print("KV_MANAGER_OK")


if __name__ == "__main__":
    main()
