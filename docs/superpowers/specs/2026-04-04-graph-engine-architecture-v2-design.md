# Graph Engine Architecture v2

## Goal

Preserve the roadmap thesis:

- chip design remains the primary validation domain
- infra remains a genuine standalone extension path, not a thin adapter
- engine stays domain-agnostic and reusable

This v2 narrows the shared core so the separation is clean in practice, not just in narrative.

## Design Principles

1. Freeze behavior contracts, not concrete inheritance trees.
2. Keep engine terminology neutral across offline and online domains.
3. Allow domains to own semantics-heavy logic: simulation, action meaning, reward shaping, and evaluation.
4. Prove extensibility with two structurally different toy domains before scaling.
5. Treat infra as a future sibling domain, not as a forced fit inside chip abstractions.

## Naming Decisions

The shared graph object should be named `ProblemGraph`.

Why:

- It represents a problem instance, not a decision tree.
- It works for both static chip netlists and dynamic scheduling snapshots.
- It avoids smuggling placement-specific semantics into the engine.

The environment object should use a process-oriented name such as `DecisionEnv` or `OptimizationEnv`.

Why:

- The env represents sequential decision-making, not the static problem structure.
- Using different prefixes for graph and env preserves the distinction between problem description and control loop.

Recommended naming set:

- `engine/contracts.py`
- `ProblemGraph`
- `ConstraintRegistry`
- `SurrogateModel`
- `DecisionEnv`
- `SearchStrategy` and `TrainingStrategy`

## Architecture Boundary

The engine owns only reusable optimization scaffolding.

Engine responsibilities:

- graph representation for problem instances
- constraint registration and reporting
- surrogate model protocol
- optimizer and trainer protocol boundaries
- experiment harness utilities that are neutral to domain semantics

Domain responsibilities:

- state semantics
- action semantics
- state transition logic
- simulator or stepper
- cost and reward shaping
- ground-truth evaluation
- domain-specific legality checks and metrics

Hard rule:

- `engine/` never imports from `chip/` or `infra/`
- domains may depend on engine contracts and helpers
- domains should satisfy engine protocols through structural typing, not inheritance

## Contract-First Core

`engine/contracts.py` is the anchor of the architecture. It should define Protocols, not ABCs.

The five required protocol boundaries are:

1. `TransitionModel`
   Turns `(state, action)` into next-step results.
   This may wrap a discrete-event simulator in infra or a placement stepper in chip.

2. `ConstraintChecker`
   Evaluates legality or rule violations and returns a structured report.

3. `Evaluator`
   Produces ground-truth domain metrics for a candidate solution or terminal state.

4. `SurrogateModel`
   Predicts proxy quality signals used for ranking, shaping, or fast search.

5. `SearchStrategy` / `TrainingStrategy`
   Encapsulates SA, analytical search, RL loops, or future optimization methods without owning domain semantics.

Suggested supporting types:

- `ProblemGraph`
- `Observation`
- `Action`
- `ConstraintViolation`
- `ConstraintReport`
- `EvaluationResult`
- `EpisodeResult`

The protocols should stay thin. If a method signature needs many chip-specific or infra-specific arguments, that is a signal the abstraction is too high-level or misplaced.

## Shared Data Model

`ProblemGraph` should capture only the reusable structural description of a problem instance.

Recommended contents:

- nodes with typed feature payloads
- edges or hyperedge encodings with typed feature payloads
- optional global features
- stable identifiers for domain objects

It should not require a shared `canvas` field.

Why:

- chip placement uses 2D geometry
- infra scheduling uses queues, machine state, time, and memory occupancy
- forcing both into one container will produce adapter-shaped code and hidden coupling

Observation and state should remain domain-defined.

Preferred approach:

- let domains pass typed dict-like objects, dataclasses, or tensors
- the engine should only require compatibility with the relevant protocol surface

## What Must Stay Out Of Engine

These belong in domains and should not migrate into `engine/`:

- OpenROAD integration
- discrete-event GPU simulation
- grid-coordinate action semantics
- job-to-machine assignment semantics
- chip reward shaping
- infra reward shaping
- DRC specifics
- KV cache semantics
- continuous batching logic

Reason:

These are exactly where domain knowledge lives. Putting them in engine would make infra look like a chip adapter or make chip inherit scheduling-shaped abstractions.

## Validation Strategy

Before scaling beyond Phase 1, the engine must pass a dual-toy validation.

Toy A: offline spatial toy

- example: place 10 rectangles on a 2D grid
- objective: minimize total pairwise distance or overlap-weighted cost
- properties: offline, spatial, no time axis

Toy B: online scheduling toy

- example: jobs arrive over time to `M` machines
- each job has size and deadline
- objective: minimize tardiness or weighted lateness
- properties: online, temporal, 1D resource assignment

Success condition:

- both toys must be implemented without modifying engine protocol definitions or shared engine APIs

This is stronger than validating with a second chip-like toy. It tests whether the extension point survives differences in time, resource model, and action semantics.

## Chip Domain Positioning

Chip remains the first real domain because it is the harder and more differentiation-rich validation case.

The chip domain should provide:

- benchmark ingestion
- graph construction from placement-relevant structures
- legality verification
- HPWL-style fast objectives
- OpenROAD-backed ground-truth evaluation
- optional RL adapter if RL remains justified after baselines

The Phase 4 terminology should be `legality verification` or `rule-aware verification`, not formal DRC.

Why:

- true DRC has a much narrower and more tool-specific meaning in EDA
- the intended scope here is placement-level legality and rule-inspired checks

## Infra Domain Positioning

Infra is still a genuine standalone extension path, but it should be framed as a future sibling domain that reuses contracts rather than a second skin on chip code.

Its depth story remains valid because it has its own hard problems:

- KV cache online paging
- continuous batching dynamics
- speculative decoding orchestration
- multi-tier model routing

That story becomes more credible when the shared layer is visibly neutral and when the first infra prototype satisfies the same contracts without forcing chip vocabulary into the design.

## Phase Gates

The roadmap should keep phases, but each phase needs explicit go/no-go criteria.

### Phase 0: Learning

Allowed output:

- notes
- diagrams
- throwaway notebooks and scripts

Not allowed:

- committing to permanent engine APIs before domain understanding is sufficient

### Phase 1 Gate

Must have:

- clean `engine/contracts.py`
- `ProblemGraph` implemented
- chip baseline working with OpenROAD-backed evaluation
- both toy domains running without engine API changes

If either toy requires engine interface changes, Phase 1 is not complete.

### Phase 2 Gate

Proxy model validation should use two thresholds:

- same-design held-out placements: target Spearman rank correlation `> 0.8`
- cross-design transfer check: target rank correlation `> 0.5`

Interpretation:

- below `0.8` on same-design holdout means the proxy is weak even as a local guide
- below `0.5` cross-design means the proxy is learning design-specific patterns rather than transferable structure

At that point, explicitly choose one path:

- accept a per-design proxy
- invest in transfer improvements
- stop treating the proxy as a reusable engine-level asset

### Phase 3 Gate

RL is not guaranteed to become the flagship method.

Proceed only if one of these is true:

- RL beats or matches heuristic baselines with a clear runtime benefit
- RL exposes a strong research story that is still publishable or interview-credible

Otherwise, keep RL as an exploratory branch rather than the center of the narrative.

### Infra Go / No-Go Gate

Infra should start only if all of the following are true:

1. dual-toy validation passed
2. chip Phase 1-3 did not force chip-specific changes into engine contracts
3. there is continued personal motivation to invest in infra as a second story

The third condition is practical, not technical, but it matters. A second project only helps if it is worth finishing.

## Testing Strategy

Testing should mirror the architecture boundary.

Engine tests:

- protocol conformance helpers
- constraint report formatting
- graph serialization or construction invariants
- search and trainer harness behavior with mocks or toys

Domain tests:

- parser and graph-building correctness
- legality checks
- evaluator integration
- simulator or stepper correctness
- reward and metric calculations

Integration tests:

- chip baseline end-to-end on a small benchmark
- toy scheduling end-to-end
- contract compatibility checks proving engine does not require domain inheritance

## Risks And Mitigations

Risk: engine abstractions become chip-shaped

Mitigation:

- use neutral names
- forbid geometry-specific required fields in shared types
- require the online toy before freezing Phase 1

Risk: engine abstractions become too generic to be useful

Mitigation:

- keep contracts behavioral and minimal
- move semantics downward into domains
- validate against real Phase 1 chip workflows, not only toy examples

Risk: infra story remains narrative-only

Mitigation:

- require the online toy scheduler as proof of extension
- defer full infra build until chip validation stabilizes the core

Risk: RL dominates the plan despite weak evidence

Mitigation:

- treat RL as contingent on baseline results
- preserve SA and analytical methods as first-class outcomes

## Recommended Repository Shape

Recommended top-level structure:

```text
engine/
  contracts.py
  graph.py
  constraints.py
  search/
  training/

chip/
  adapters/
  evaluators/
  legality/
  parsers/
  experiments/

infra/
  adapters/
  simulation/
  evaluators/
  policies/
  experiments/

toys/
  spatial/
  online_scheduler/
```

This layout makes the architecture intent visible before the project becomes large.

## Final Recommendation

Keep the roadmap thesis, but update the execution spec as follows:

- preserve chip-first sequencing
- preserve infra as a real future sibling domain
- replace placement-centric shared abstractions with neutral contracts
- anchor the whole design in `engine/contracts.py`
- validate extensibility with both an offline spatial toy and an online scheduling toy
- use explicit phase gates so infra and RL remain earned expansions, not assumptions

This keeps the ambition while making the architecture much more defensible and much more likely to survive first contact with implementation.
