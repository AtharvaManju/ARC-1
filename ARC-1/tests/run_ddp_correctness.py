import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(1024, 1024, bias=False)
    def forward(self,x):
        return self.l(x).relu()

def worker(rank, world, pool_dir):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world)

    torch.cuda.set_device(rank)
    m = Tiny().cuda()
    ddp = torch.nn.parallel.DistributedDataParallel(m, device_ids=[rank])
    opt = optim.AdamW(ddp.parameters(), lr=1e-3)

    cfg = AIMemoryConfig(pool_dir=pool_dir, spill_min_bytes=1*1024*1024, rank=-1, world_size=-1, sync_each_step=True)
    ctrl = AIMemoryController(cfg)

    x = torch.randn(16, 1024, device="cuda", dtype=torch.float16, requires_grad=False)

    with ctrl.step(profiling_warmup=True):
        y = ddp(x)
        y.float().mean().backward()
    ctrl.finalize_pcc_profile()

    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(16, 1024, device="cuda", dtype=torch.float16)
        with ctrl.step():
            y = ddp(x)
            y.float().mean().backward()
        opt.step()

    ctrl.shutdown()
    dist.destroy_process_group()

def main():
    if torch.cuda.device_count() < 2:
        raise SystemExit("Need 2 GPUs for this test")
    pool_dir = "/mnt/nvme_pool"
    mp.spawn(worker, args=(2, pool_dir), nprocs=2, join=True)
    print("DDP OK")

if __name__ == "__main__":
    main()
