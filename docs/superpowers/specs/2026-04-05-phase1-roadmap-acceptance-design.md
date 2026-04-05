# Phase 1 Roadmap Acceptance Design

## Goal

Upgrade the current engineering-complete Phase 1 into a roadmap-acceptable Phase 1.

For this spec, "Phase 1 complete" means satisfying the original roadmap intent in [graph_placement_engine_roadmap.md](C:/Users/14574/Chip/graph_placement_engine_roadmap.md):

- run a simulated-annealing baseline on three public ISPD benchmarks across a small/medium/larger scale spread
- evaluate our SA placement and OpenROAD's built-in placer through the same OpenROAD tail flow
- produce a reproducible baseline results table with routed and timing metrics, not just parser-level HPWL

## Current Gaps

The repo already has:

- neutral engine primitives
- chip parser, cost, constraints, and SA baseline
- Dockerized OpenROAD access
- real-design export and reporting scaffolding

What it does not yet have is the exact acceptance path the roadmap promises:

- three public ISPD benchmarks as the benchmark set
- a true placement injection mechanism inside OpenROAD
- a fair one-template evaluation flow for both built-in placement and SA placement
- a final comparison table with routed wirelength, timing, and OpenROAD-reported DRC sanity checks

## Scope

This acceptance slice is intentionally narrow.

It includes:

1. benchmark selection and cataloging for three public ISPD designs
2. OpenROAD placement injection for externally generated placements
3. one shared evaluation flow template for built-in and SA placements
4. an acceptance runner that emits the final Phase 1 comparison table

It does not include:

- Phase 2 proxy-model work
- Phase 3 RL work
- a second probe domain
- reworking the engine public API
- a new custom DEF writer as the primary path

## Benchmark Policy

The benchmark set must be public ISPD benchmarks, but selection should optimize for stable execution rather than prestige.

Required spread:

- one small benchmark in the `ibm01` scale class
- one medium benchmark in the `adaptec1` scale class
- one somewhat larger benchmark in the same family or an adjacent public ISPD family

Selection rule:

- prefer the first three candidates that have a stable OpenROAD-compatible ingestion path
- if a candidate benchmark consumes more than half a day in format or tool compatibility debugging, replace it with another benchmark in the same scale bucket

Practical interpretation:

- the acceptance target is "three public ISPD designs with small/medium/larger scale coverage"
- it is not "make one specific troublesome benchmark work at all costs"

## Acceptance Architecture

### 1. Benchmark Catalog Layer

Add a benchmark manifest module that records:

- benchmark name
- source family
- scale bucket
- local artifact paths
- ingestion mode
- compatibility status

The manifest should be the only place that knows which exact three designs make up the acceptance set.

This lets us swap one benchmark for another within the same scale bucket without touching evaluation logic.

### 2. Placement Injection Strategy

Primary strategy: inject placements inside OpenROAD via Tcl `place_inst` commands.

Why this is the default:

- OpenROAD owns internal state consistency
- we avoid hand-writing DEF placement records
- we avoid subtle mismatches around orientation, fixed/movable state, and placement semantics

Required behavior:

- load the design in OpenROAD
- identify movable instances to update
- apply `(x, y, orient)` placements through Tcl
- preserve fixed instances unless the benchmark mode explicitly allows changing them

Fallback policy:

- only build a DEF rewrite fallback if `place_inst` proves insufficient for a benchmark class
- treat DEF writing as a contingency path, not the primary implementation

### 3. Single-Template Evaluation Flow

`openroad_eval.py` should own one canonical OpenROAD flow template.

The only difference between evaluation modes should be the placement injection point:

- `built_in` mode
  - run OpenROAD's own placement stage
- `sa_injected` mode
  - skip built-in placement
  - inject our external placement

Everything after that point must be identical:

- the same legalization or detailed-placement policy if used
- the same routing stage
- the same timing estimation commands
- the same metric extraction

This is the fairness rule for the acceptance run. No separate flow scripts for built-in and SA.

### 4. Acceptance Runner

Add one experiment entrypoint that:

- iterates over the three accepted ISPD benchmarks
- runs the SA baseline
- evaluates the SA placement through the shared OpenROAD flow
- evaluates the built-in placer through the same flow
- emits one final comparison artifact

The artifact should be machine-readable first. Human-readable formatting can be layered on top afterward.

## Metrics

The acceptance results table must include, at minimum:

- benchmark name
- scale bucket
- node count
- net count
- legality status before OpenROAD evaluation
- SA runtime
- built-in placer runtime
- HPWL before OpenROAD flow
- total routed wirelength
- WNS
- TNS
- OpenROAD-reported DRC violation count

The routed metric should be explicitly `total routed wirelength`, not a vague routed score.

The DRC count is a sanity-check metric:

- if SA produces DRC violations far above the built-in placer, the benchmark run should not be treated as a pass, even if HPWL looks competitive

## Data Flow

For each benchmark:

1. benchmark catalog resolves source artifacts
2. chip parser builds `ProblemGraph` and `ChipProblem`
3. SA produces an external placement
4. OpenROAD evaluation runs twice through the same template:
   - once with built-in placement
   - once with injected SA placement
5. metric extractor normalizes both runs into one row
6. acceptance runner writes the full results table

## Engine Boundary

This acceptance work should not widen the shared engine contract.

Phase 1 gate wording for this slice:

- chip-domain acceptance work must not require changes to `engine/` public APIs for chip-specific reasons

The cross-domain gate involving a second probe domain stays out of scope for this acceptance push.

Rationale:

- the online toy scheduler is valuable, but it is not required to satisfy the roadmap's original Phase 1 wording
- trying to satisfy both the chip acceptance path and the cross-domain gate in one slice would blur scope again

## Error Handling

The acceptance runner should degrade explicitly and keep failure reasons visible.

Examples:

- benchmark incompatible with chosen ingestion path
- OpenROAD placement injection failure
- routing failure
- timing report missing
- DRC report unavailable

Rows should surface failure modes rather than silently dropping metrics.

## Testing Strategy

Testing should split into three levels.

Unit tests:

- benchmark manifest selection
- placement injection script generation
- metric parsing
- results table assembly

Small integration tests:

- OpenROAD flow template generation for both placement modes
- injected-placement path uses the same tail commands as built-in mode

Manual acceptance verification:

- run the full three-benchmark acceptance command
- save the produced comparison table under `experiments/chip/`

The full acceptance command is too expensive to make part of ordinary `pytest`, but it must be reproducible from one documented entrypoint.

## Success Criteria

Phase 1 is accepted against the roadmap when all of the following are true:

1. `pytest -v` passes on the branch implementing this slice
2. the accepted benchmark set contains three public ISPD designs across small/medium/larger scale buckets
3. SA placement is applied inside OpenROAD through the primary `place_inst` path
4. built-in and SA evaluation use the same OpenROAD tail flow
5. the final table reports `total routed wirelength`, `WNS`, `TNS`, and OpenROAD-reported `DRC violation count`
6. the table is generated from one command without manual post-processing

## Non-Goals

- no attempt to complete Phase 2-5
- no engine redesign
- no cross-domain acceptance gate
- no benchmark heroics on a single broken ISPD design

## Recommendation

Implement this as a strict acceptance overlay on top of the current Phase 1 base:

- select stable public ISPD benchmarks by scale bucket
- inject SA placements through Tcl `place_inst`
- evaluate both placement modes through one OpenROAD flow template
- publish one reproducible comparison table

That gives us the narrowest path from "engineering-complete Phase 1" to "roadmap-acceptable Phase 1" without re-expanding scope in unrelated directions.
