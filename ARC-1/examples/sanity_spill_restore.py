import torch
from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

def main():
    cfg = AIMemoryConfig(
        pool_dir="/mnt/nvme_pool",
        spill_min_bytes=1 * 1024 * 1024,
        encrypt_at_rest=True,
        encryption_key_path="/mnt/nvme_pool/test_key.key",
        sync_each_step=True,
    )
    ctrl = AIMemoryController(cfg)

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)

    with ctrl.step(profiling_warmup=True):
        y = (x @ x).sum()
        y.backward()

    ctrl.finalize_pcc_profile()

    x2 = torch.randn(1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)
    with ctrl.step():
        y2 = (x2 @ x2).sum()
        y2.backward()

    print("OK", ctrl.quick_summary())
    ctrl.shutdown()

if __name__ == "__main__":
    main()
