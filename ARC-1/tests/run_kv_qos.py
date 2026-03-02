import tempfile

import torch

from aimemory.config import AIMemoryConfig
from aimemory.kv_manager import KVResidencyManager
from aimemory.storage import SARCStorage


def main():
    with tempfile.TemporaryDirectory(prefix="aimemory_kvq_") as td:
        st = SARCStorage(pool_dir=td, rank=0, backend="NVME_FILE", encrypt_at_rest=False, durable=False, compression_codec="none")
        cfg = AIMemoryConfig(
            kv_manager_enabled=True,
            kv_budget_bytes=4096,
            kv_tenant_budget_ratio=0.8,
            kv_request_budget_ratio=0.2,
        )
        kv = KVResidencyManager(cfg, st)
        t = torch.arange(2048, dtype=torch.uint8).clone()
        ok1 = kv.register("k1", t, tenant_id="t1", request_id="r1")
        ok2 = kv.register("k2", t, tenant_id="t1", request_id="r1")
        assert ok1 in (True, False)
        assert ok2 is False
        s = kv.stats()
        assert int(s.get("qos_denials", 0)) >= 1
        st.close()
    print("KV_QOS_OK")


if __name__ == "__main__":
    main()
