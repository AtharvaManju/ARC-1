# ARC-1 GA Checklist

## Runtime
- Qualification report exists and `passed=true`.
- Fleet `safe_mode_ranks=0` in canary window.
- Fleet `rank_skew_pct <= 25`.
- No integrity quarantine events.

## Rollout Safety
- Policy is signed and versioned.
- Canary ratio staged (`1% -> 10% -> 50% -> 100%`).
- Auto-rollback thresholds configured.
- Agent health endpoints wired to orchestration.

## Operations
- Support bundle procedure documented.
- Disk quota and retention policies configured.
- On-call runbook includes fail-open behavior.

## Acceptance
- p95/p99 overhead within agreed SLO.
- Training outcome parity checks pass.
- Compile/capture parity checks pass on target stack.
