# ARC-1 Security Runbook

## Threat Model
- Generate baseline threat model artifact:
  - `arc1 security-threat-model --out ./arc1_threat_model.json`

## Security Audit
- Run filesystem/key/audit checks:
  - `arc1 security-audit --pool-dir /mnt/nvme_pool --key-path /path/to/key --audit-path /path/to/audit.log --out ./arc1_security_audit.json`

## Key Rotation
- Rotate key and emit audit event:
  - `arc1 security-rotate-key --key-path /path/to/key --new-key-path /path/to/new_key --audit-path /path/to/audit.log`

## Incident Response
1. Set spill fail-open policies if corruption or IO faults are observed.
2. Collect support bundle and policy/fleet reports.
3. Quarantine affected namespace and rotate keys if compromise is suspected.
4. Re-run qualification gates before restoring rollout stage.
