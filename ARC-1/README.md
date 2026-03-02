# AIMemory v0.10.2 — Plug-and-Play Tiers

AIMemory reduces effective VRAM usage by spilling large autograd saved tensors.

## Tiers
- **Tier 0 (Always works):** CPU-only imports and runs in **NOOP** mode.
- **Tier 1 (CUDA, no NVMe/engine):** **RAM backend** (spills to pinned CPU pool).
- **Tier 2 (CUDA + NVMe):** **NVME_FILE backend**.
- **Tier 2b (CUDA + NVMe + engine wheel):** best performance (native IO + aligned pinned alloc).

## Install
Base (works everywhere):
```bash
pip install aimemory
```

Enable dashboard:
```bash
pip install "aimemory[dashboard]"
```

Install engine:
```bash
pip install aimemory-engine
```

## Important perf note
By default, AIMemory does **not** call `torch.cuda.synchronize()` each step.
Step timing uses **CUDA events** (stream-local wait) to avoid destroying overlap.
If you want a full device sync each step (debug/bench), set:
```py
cfg.sync_each_step = True
```

## Quick Start
```py
from aimemory.config import AIMemoryConfig
from aimemory.controller import AIMemoryController

cfg = AIMemoryConfig(pool_dir="/mnt/nvme_pool")
ctrl = AIMemoryController(cfg)

with ctrl.step(profiling_warmup=True):
    forward(); loss.backward()
ctrl.finalize_pcc_profile()

for _ in range(1000):
    with ctrl.step():
        forward(); loss.backward()
```

## Ops
- `aimemory doctor`
- `aimemory gc`
- `aimemory support-bundle --out bundle.zip`
- `aimemory headroom-gate --pool-dir /mnt/nvme_pool --out ./aimemory_headroom_gate.json`
- `aimemory qualify --pool-dir /mnt/nvme_pool --out ./aimemory_qualification.json`
- `aimemory consistency-check --pool-dir /mnt/nvme_pool --rank 0`
- `aimemory policy-push --pool-dir /mnt/nvme_pool --name prod --file policy.json`
- `aimemory fleet-report --pool-dir /mnt/nvme_pool`
- `aimemory agent --bind 0.0.0.0 --port 9765`
