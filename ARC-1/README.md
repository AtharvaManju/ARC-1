# ARC-1 v0.10.2 — Plug-and-Play Tiers

ARC-1 reduces effective VRAM usage by spilling large autograd saved tensors.

## Tiers
- **Tier 0 (Always works):** CPU-only imports and runs in **NOOP** mode.
- **Tier 1 (CUDA, no NVMe/engine):** **RAM backend** (spills to pinned CPU pool).
- **Tier 2 (CUDA + NVMe):** **NVME_FILE backend**.
- **Tier 2b (CUDA + NVMe + engine wheel):** best performance (native IO + aligned pinned alloc).

## Install
Base (works everywhere):
```bash
pip install arc1
```

Enable dashboard:
```bash
pip install "arc1[dashboard]"
```

Install engine:
```bash
pip install arc1-engine
```

## Important perf note
By default, ARC-1 does **not** call `torch.cuda.synchronize()` each step.
Step timing uses **CUDA events** (stream-local wait) to avoid destroying overlap.
If you want a full device sync each step (debug/bench), set:
```py
cfg.sync_each_step = True
```

## One-Line Start
```py
import arc1
arc1.enable()
```

## Advanced Start
```py
from arc1 import ARC1Config, ARC1Controller

cfg = ARC1Config(pool_dir="/mnt/nvme_pool")
ctrl = ARC1Controller(cfg)

with ctrl.step(profiling_warmup=True):
    forward(); loss.backward()
ctrl.finalize_pcc_profile()

for _ in range(1000):
    with ctrl.step():
        forward(); loss.backward()
```

## Ops
- `arc1 doctor`
- `arc1 gc`
- `arc1 support-bundle --out bundle.zip`
- `arc1 headroom-gate --pool-dir /mnt/nvme_pool --out ./aimemory_headroom_gate.json`
- `arc1 qualify --pool-dir /mnt/nvme_pool --out ./aimemory_qualification.json`
- `arc1 consistency-check --pool-dir /mnt/nvme_pool --rank 0`
- `arc1 policy-push --pool-dir /mnt/nvme_pool --name prod --file policy.json`
- `arc1 fleet-report --pool-dir /mnt/nvme_pool`
- `arc1 agent --bind 0.0.0.0 --port 9765`
- `arc1 ga-readiness --pool-dir /mnt/nvme_pool --qualification ./aimemory_qualification.json`
- `arc1 commercial-pack --pool-dir /mnt/nvme_pool --qualification ./aimemory_qualification.json --out-dir ./arc1_pack`
- `arc1 compile-matrix --out ./arc1_compile_matrix.json`
- `arc1 parity-longrun --pool-dir /mnt/nvme_pool --out ./arc1_parity_longrun.json`
- `arc1 fastpath-qualify --pool-dir /mnt/nvme_pool --out ./arc1_fastpath_qualification.json`
- `arc1 migration-report --path . --out ./arc1_migration_report.json`
- `arc1 security-threat-model --out ./arc1_threat_model.json`
- `arc1 security-audit --pool-dir /mnt/nvme_pool --out ./arc1_security_audit.json`
- `arc1 claims-evidence --qualification ./aimemory_qualification.json --fastpath ./arc1_fastpath_qualification.json --out ./arc1_claims_evidence.json`
