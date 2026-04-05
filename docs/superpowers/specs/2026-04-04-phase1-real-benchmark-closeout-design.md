# Phase 1 Real Benchmark Closeout Design

## Goal

Close Phase 1 as an engineering-complete baseline phase by producing a repeatable real-benchmark workflow for three OpenROAD-flow designs: `gcd`, `aes`, and `ibex`.

This closeout is intentionally narrower than a full research evaluation. It should prove that the engine and chip domain can ingest realistic placement outputs, run the internal baseline stack safely, and summarize comparable benchmark metrics without manually stitching commands together.

## Scope

This slice includes three outcomes:

1. export real ORFS placement-stage artifacts into a repo-local format the chip domain already understands
2. make the baseline experiment robust on realistic designs where the analytical relaxation is too slow to be practical
3. generate a benchmark summary table with metrics that are honest, reproducible, and available today

This slice explicitly does not include:

- writing our placement back into DEF and re-running routed evaluation in OpenROAD
- extending Phase 1 into STA-accurate or DRC-accurate research evaluation
- broadening the engine API surface

## Why This Is The Right Phase 1 Finish

The roadmap Phase 1 asks for a working baseline on multiple benchmarks. Right now the project has the right architectural pieces, but the final benchmark story still depends on partial manual steps and a toy benchmark.

This design finishes Phase 1 without silently turning it into a larger systems project. The output should be a credible baseline table and a reusable workflow, not a half-built placement reinjection system.

## Architecture Additions

### 1. Real Benchmark Export Utility

Add a small chip-domain utility that exports placement-stage artifacts from ORFS outputs into a stable local directory layout:

- input: `orfs-work/results/nangate45/<design>/base/3_place.odb`
- output directory: `orfs-export/nangate45/<design>/`
- output files:
  - `3_place.def`
  - `3_place.sdc` when present
  - `NangateOpenCellLibrary.tech.lef`
  - `NangateOpenCellLibrary.macro.mod.lef`

This utility should use Dockerized OpenROAD because that is the supported installation path in the current environment.

The exporter belongs in `chip/`, not `engine/`, because it is EDA-tool-specific orchestration rather than shared optimization logic.

### 2. Large-Design-Safe Baseline Path

`chip/experiments/baseline.py` should keep its existing small-design behavior, but it needs a safe mode for realistic exported designs.

Required behavior:

- always compute `initial`
- always run `SA`
- run `analytical` only when the graph size is under a configurable threshold
- when analytical is skipped, return structured metadata explaining why it was skipped

This keeps the baseline harness usable on real designs without pretending that the current `cvxpy` relaxation is production-scalable.

### 3. Benchmark Summary Reporter

Add a narrow experiment/report path that:

- iterates over `gcd`, `aes`, and `ibex`
- ensures export artifacts exist
- parses each exported design
- runs the safe baseline experiment
- emits a machine-readable summary that includes:
  - design name
  - node count
  - fixed node count
  - net count
  - initial HPWL
  - SA HPWL
  - legality for initial and SA placements
  - analytical status
  - OpenROAD reference availability

The reporter can live in `chip/experiments/` and write output under `experiments/chip/` or print JSON for now. The key requirement is repeatability, not polish.

## Data Flow

1. ORFS produces `3_place.odb`
2. exporter materializes a DEF/LEF bundle under `orfs-export/`
3. existing chip parser loads the exported design into `ProblemGraph` and `ChipProblem`
4. baseline harness runs safe evaluation
5. reporter aggregates per-design metrics into one summary artifact

This keeps the current parser and baseline logic as the center of gravity, which is the right architecture for Phase 1.

## Error Handling

The new path should degrade explicitly instead of failing opaquely.

Expected degradations:

- missing Docker or missing image: exporter reports tool unavailable
- missing `3_place.odb`: reporter marks design unavailable and continues
- missing `3_place.sdc`: export succeeds without SDC
- analytical over size threshold: baseline reports `skipped` with reason
- OpenROAD evaluation failures inside the existing bridge: baseline keeps results and captures stderr

The reporter should never hide skipped work. Every missing metric should have a reason.

## Testing Strategy

Tests should stay focused and cheap:

- exporter unit test covering command construction and expected output paths
- baseline regression test covering analytical skip behavior on large graphs
- reporter test covering mixed available/missing designs and summary shape

No test should require running a multi-minute ORFS flow. Integration with real exported designs can be verified through direct commands during manual validation.

## Success Criteria

Phase 1 closeout is successful when all of the following are true:

1. `pytest -v` passes fresh in the worktree
2. `gcd`, `aes`, and `ibex` can each be exported from existing ORFS placement-stage outputs into `orfs-export/`
3. the baseline harness completes on all three exported designs without hanging on analytical placement
4. a single summary artifact or command output shows the real-design baseline table

## Non-Goals

- no new shared engine abstractions
- no infra work
- no RL work
- no placement reinjection into OpenROAD
- no claims about routed wirelength for our custom placements

## Final Recommendation

Treat this as a narrow engineering closeout for Phase 1:

- automate export
- make baseline execution safe on realistic designs
- publish a reproducible benchmark summary

That gives the project an honest “Phase 1 passed” story without overreaching beyond what the current system actually supports.
