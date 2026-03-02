# ARC-1 Migration Guide (`aimemory` -> `arc1`)

## Package Names
- Core package: `arc1`
- Engine package: `arc1-engine`

Compatibility aliases remain available:
- Python module alias: `aimemory` (legacy)
- CLI alias: `aimemory` (legacy)

## Code Changes
- Replace:
  - `import aimemory` -> `import arc1`
  - `from aimemory...` -> `from arc1...` where possible
- One-line integration:
  - `import arc1; arc1.enable()`

## Tooling
- Generate migration inventory:
  - `arc1 migration-report --path <repo>`
- Optional automated rewrite:
  - `arc1 migration-report --path <repo> --rewrite --out migration.json`

## Rollout Strategy
1. Ship with alias compatibility enabled.
2. Run migration report in all repos.
3. Switch CI/docs/examples to `arc1`.
4. Keep aliases during deprecation window.
