import torch
import torch.nn as nn
import torch.optim as optim

from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8192, 8192, bias=False).cuda()
        self.l2 = nn.Linear(8192, 8192, bias=False).cuda()

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

def main():
    m = Toy().cuda()
    opt = optim.AdamW(m.parameters(), lr=1e-3)

    cfg = AIMemoryConfig(pool_dir="/mnt/nvme_pool", spill_min_bytes=8 * 1024 * 1024)
    ctrl = AIMemoryController(cfg)

    x = torch.randn(4, 8192, device="cuda")
    with ctrl.step(profiling_warmup=True):
        y = m(x)
        loss = (y * y).mean()
        loss.backward()
    ctrl.finalize_pcc_profile()

    for step in range(20):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(4, 8192, device="cuda")
        with ctrl.step():
            y = m(x)
            loss = (y * y).mean()
            loss.backward()
        opt.step()
        if step % 5 == 0:
            print(ctrl.quick_summary())

if __name__ == "__main__":
    main()
