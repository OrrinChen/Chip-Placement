# Phase 3 Macroplacement Benchmark Switch Design

## Goal

Replace the current `mgc_*` Phase 3 real-chip evaluation path with true macro-placement benchmarks, starting with `Ariane` from the TILOS MacroPlacement benchmark suite.

## Why This Switch Is Necessary

The current `mgc_*` real-chip benchmarks are valid for:

- Phase 1 acceptance
- Phase 2 proxy validation

They are not valid as Phase 3 macro-placement RL benchmarks because they are standard-cell-dominant designs with tens of thousands of movable cells, not macro-dominant designs.

That creates a formulation mismatch:

- the current sequential masked-grid PPO setup is intended for macro placement
- the `mgc_*` designs behave like standard-cell placement
- full-design legality masking becomes intractable under the current action formulation

This is a benchmark/formulation issue, not a data-quality issue.

## Scope

This slice is intentionally narrow.

It does not attempt to complete Phase 3 on the final benchmark set immediately.

It only answers the first hard gate:

1. Can `Ariane` be ingested and evaluated with the existing LEF/DEF/OpenROAD infrastructure?
2. If yes, is full-design macro-placement PPO feasible under the current legality-check implementation?

## Benchmark Strategy

Start with exactly one benchmark:

- `Ariane`

Do not expand to `BlackParrot` or `MemPool` until `Ariane` passes both:

- pure OpenROAD evaluation sanity
- full-design legality/per-step cost sanity

If `Ariane` fails either gate, stop and record the blocker before widening the benchmark set.

## Success Criteria For This Slice

### Gate A: Benchmark Ingestion

For `Ariane`, the codebase must be able to:

- locate or fetch the benchmark assets
- parse the benchmark into `ChipProblem`
- run a pure OpenROAD evaluation on a reference DEF
- collect `HPWL`, `routed wire length`, `WNS`, `TNS`, `DRC violations`, and runtime

### Gate B: PPO Feasibility Probe

For `Ariane`, measure:

- movable macro count
- single legality-check latency
- full `build_action_mask()` latency at `32x32`

Decision rule:

- if legality check is comfortably within a PPO-viable range, proceed to full-design PPO bring-up
- if legality check is too slow, mark the blocker explicitly as legality-performance, not PPO-model failure

## Required Output

This slice must produce:

- one canonical Ariane asset manifest / probe artifact
- one Ariane evaluation artifact
- one explicit feasibility note stating one of:
  - `Ariane full-design PPO feasible under current legality path`
  - `Ariane full-design PPO blocked by legality performance`

## Architectural Rule

Do not delete or rewrite the existing `mgc_*` Phase 1 / Phase 2 paths.

Those remain valid for:

- Phase 1 acceptance
- Phase 2 proxy training/evaluation

This new benchmark switch is Phase 3-specific.
