# Phase 3 PPO Model Design

## Goal

Define the first production-worthy PPO model design for roadmap Phase 3 chip placement RL.

This spec covers the PPO model and its immediate training interface, not the full training loop implementation. The goal is to lock the state/action/reward contract now so later PPO work does not drift into reward bugs, action-space churn, or silent data leakage.

## Design Constraints

This PPO design is intentionally narrow.

It assumes:

- fixed macro placement order
- masked discrete grid actions
- single-design training for the first training runs
- reward driven by the Phase 2 frozen linear proxy plus analytic HPWL shaping
- no orientation head in v1

It explicitly does not include:

- joint macro-selection plus position-selection
- continuous-action PPO
- multi-design PPO training
- OpenROAD-in-the-loop per-step rewards
- curriculum scheduling details beyond the initial small-benchmark-first recommendation

## Why This Design

For PPO-based macro placement, the mainstream RL formulation is a sequential MDP where the policy places the current macro on a discrete grid rather than jointly deciding both macro order and location. This is the same broad framing used in the Google/AlphaChip line of work and its follow-on papers, even though the overall placement field is still dominated by analytical and heuristic placers rather than PPO.

Relevant references:

- Nature 2021 AlphaChip paper: [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w)
- Google open-source Circuit Training: [google-research/circuit_training](https://github.com/google-research/circuit_training)
- Follow-on rectilinear macro placement RL work: [Toward RL-based Rectilinear Macro Placement](https://arxiv.org/abs/2311.03383)

For this project, the design is even narrower than those papers because the current codebase already has:

- a validated Phase 2 frozen linear proxy on full placements
- a row-aware chip adapter and environment shell
- a strong lesson from Phase 2 that simple, well-conditioned models beat higher-capacity learned heads when data is still small

That makes the right v1 choice:

- fixed order
- masked `32 x 32` grid policy
- same-design PPO
- seed-completed proxy reward

## Fixed Placement Order

The v1 PPO policy does not choose which macro to place next.

Placement order is deterministic:

1. sort by descending macro area
2. break ties by macro name using plain string lexicographic order

This order must be implemented exactly as `(-area, name)` and must not depend on hash iteration order or any unstable container semantics.

Rationale:

- it keeps the PPO action space small and learnable
- it matches the standard sequential macro-placement PPO framing
- it preserves reproducibility across machines and runs

Important caveat:

- fixed order works here because the reward design follows the same general cheap-cost-sequential-placement logic as the AlphaChip family
- this should not be treated as a universal theorem that order never matters

## Grid Resolution

The v1 policy uses a `32 x 32` grid.

That means:

- action count = `1024`
- policy logits shape = `[batch, 1024]`

Why `32 x 32`:

- small enough to make masked PPO bring-up practical
- large enough to express meaningful coarse placement decisions
- avoids starting at `64 x 64` or `128 x 128`, where action-head size and masking/debugging cost jump significantly

Planned upgrade path:

- `v1`: `32 x 32`
- `v2`: `64 x 64` only after PPO bring-up is stable and action masking is verified

## Observation Schema

The PPO model consumes three primary branches plus one mask.

### 1. Graph Branch

Source:

- `observation["graph"]` from `chip.rl_adapter.ChipRLAdapter.observe`

Representation:

- node features: `float32`, shape `[N, F_node]`
- edge index: `int64`, shape `[2, E]`
- optional graph-level features: `float32`, shape `[F_graph]`

Required node features in v1:

- macro geometry: `width`, `height`
- current coordinates: `x`, `y`
- fixed flag
- macro kind
- orientation
- placed flag
- current macro flag

The `placed flag` and `current macro flag` are new PPO-facing fields and should be added explicitly by the adapter rather than inferred downstream.

### 2. Current-Macro Branch

Source:

- the current macro is determined by the fixed order and current step index

Representation:

- one node embedding extracted from the graph branch for the current macro
- shape after graph encoder: `[batch, D_model]`

This branch exists because the policy must choose a position conditioned on the identity and geometry of the next macro to place, not just on the global graph state.

### 3. Canvas Branch

Source:

- adapter-generated grid tensor aligned to the `32 x 32` action lattice

Representation:

- `float32`, shape `[batch, C, 32, 32]`

Required v1 channels:

- occupancy mask
- fixed-block mask
- placed-macro mask
- current legal-placement mask before action masking
- normalized x coordinate
- normalized y coordinate

Optional later channels:

- coarse density map
- congestion prior
- timing-criticality prior

### 4. Action Mask

Source:

- adapter-generated legality mask for the current macro on the current grid

Representation:

- boolean mask, shape `[batch, 32, 32]`
- `True` means invalid action
- `False` means valid action

This mask is separate from the canvas tensor because it must feed directly into masked softmax.

## Encoder Structure

The PPO trunk uses three encoders.

### Graph Encoder

Use a GraphSAGE-style encoder aligned with the existing `engine.gnn_proxy` direction and the AlphaChip-style GraphSAGE family, rather than introducing a new edge-specific GNN at v1.

Output:

- per-node embeddings: `[batch_nodes, D_model]`
- pooled global graph embedding: `[batch, D_model]`

Why GraphSAGE in v1:

- implementation is already close to the current repo's GNN patterns
- lower engineering risk than introducing a new hypergraph/edge-GNN stack immediately
- enough to validate the PPO architecture before revisiting graph operator choice

### Canvas Encoder

Use a small CNN over the `32 x 32` canvas tensor.

Recommended v1 shape:

- 3 conv blocks
- stride pattern preserving enough spatial resolution for action logits
- output tensor shape `[batch, D_canvas, 32, 32]`

This branch should stay spatial; do not fully pool it before the policy head.

### Current-Macro Encoder

No separate tower in v1.

Instead:

- use the graph encoder's node embedding for the current macro
- concatenate the raw scalar macro geometry features if needed

This keeps the identity of the current item tightly coupled to the graph branch.

## Fusion Choice

Use **FiLM-style conditioning** of the canvas branch by the graph branch.

Specifically:

- build a conditioning vector from:
  - current macro embedding
  - global graph embedding
- pass that vector through a small MLP to produce per-channel scale and bias
- modulate canvas features channel-wise

Chosen over concat-broadcast because:

- concat-broadcast inflates spatial feature maps and head parameters
- v1 should preserve small-model stability
- FiLM is a clean way to let the current macro and global graph state condition the spatial scoring function without exploding parameter count

The value head will use pooled representations, but the policy head must keep the spatially conditioned canvas map.

## Policy Head

The policy head outputs logits over `32 x 32` grid cells.

Structure:

1. FiLM-conditioned canvas tensor `[batch, D_canvas, 32, 32]`
2. `1x1` or small-kernel conv projection to one logit per cell
3. flatten to `[batch, 1024]`

### Masked Softmax

Illegal actions must be masked with actual negative infinity before softmax:

```python
masked_logits = logits.masked_fill(action_mask.view(batch, -1), float("-inf"))
policy = torch.softmax(masked_logits, dim=-1)
```

Do not use large negative constants like `-1e9` as the canonical implementation. `float("-inf")` should be the default because it is safer as grid size grows.

## Value Head

The value head shares the trunk but not the spatial action projection.

Inputs:

- global graph embedding
- current macro embedding
- pooled canvas embedding

Fusion:

- concatenate the three vectors
- feed through a small MLP
- output one scalar value per state

Output shape:

- `[batch, 1]`

The value head should not reuse the policy logits directly.

## Reward Function

The reward must be well-defined on partial placements.

The Phase 2 proxy was trained on full placements, not partial ones, so v1 PPO must use **seed completion**.

### Seed Completion

For any partial state:

- already placed macros use their current PPO positions
- unplaced macros are filled with their seed DEF positions

This produces a pseudo-full placement that can be scored by the Phase 2 frozen linear proxy.

### Dense Reward Terms

At step `t`, define:

- `P_t`: seed-completed placement after action `t`
- `P_{t-1}`: seed-completed placement before action `t`
- `Proxy(P)`: frozen linear Phase 2 proxy prediction on completed placement `P`
- `HPWL(P)`: analytic HPWL on completed placement `P`

Dense reward:

`r_t = w_proxy * DeltaProxy_t + w_hpwl * DeltaHPWL_t - w_legality * I_t`

where:

- `DeltaProxy_t = -(Proxy(P_t) - Proxy(P_{t-1}))`
- `DeltaHPWL_t = -(HPWL(P_t) - HPWL(P_{t-1}))`
- `I_t = 1` if the post-action state is illegal, else `0`

Recommended v1 weights:

- `w_proxy = 1.0`
- `w_hpwl = 0.25`
- `w_legality = 5.0`

### Proxy Stabilization

Use `proxy_delta`, not raw `proxy_value`, as the proxy term.

In addition, normalize proxy deltas within each rollout batch using running standard deviation:

- do not mean-center in v1
- divide by `max(running_std, epsilon)`

This is the chosen mitigation for weak early-episode signal under seed completion. Do not introduce step-dependent proxy-weight schedules in v1.

### Terminal Reward

Terminal reward in v1 remains cheap:

- reuse the same dense terms
- add a larger terminal legality penalty if the final state is invalid

Do not call OpenROAD inside the PPO inner training loop. OpenROAD stays in evaluation only.

## Training Scope

The first PPO training runs are **single-design only**.

This follows directly from Phase 2:

- same-design proxy signal is strong
- cross-design proxy signal is materially weaker

So v1 PPO should train and evaluate within one benchmark at a time, starting from the smallest stable benchmark.

Recommended order:

1. `mgc_pci_bridge32_1`
2. `mgc_pci_bridge32_2`
3. `mgc_des_perf_2`

## Baselines Before PPO

PPO should not be the first algorithm to touch this action space and reward.

Two bring-up baselines are required first:

### 1. Masked Random

- samples only valid actions
- establishes the floor for episode return and final HPWL/proxy score

### 2. Greedy One-Step Lookahead

- evaluates candidate valid cells using **analytic HPWL delta only**
- chooses the best immediate move for the current macro

This greedy baseline is intentionally not driven by the proxy term. That makes it a more independent ceiling for PPO bring-up.

SA and analytical placement remain important final report baselines, but they are not the first PPO bring-up baselines.

## Required Adapter Changes

`chip/rl_adapter.py` must grow a few PPO-facing outputs.

Add to `observe(...)`:

- `current_macro`: macro name
- `current_macro_index`: index in fixed order
- `placed_count`
- `graph`
- `canvas`
- `action_mask`
- `placement`
- `movable_macros`
- `hpwl`

Add adapter helpers:

- `ordered_macros()` returning the fixed `(-area, name)` sequence
- `current_macro_for_step(step_index)`
- `build_canvas(state, step_index, grid_shape)`
- `build_action_mask(state, step_index, grid_shape)`
- `seed_completed_placement(state, step_index)`
- `proxy_delta(previous_state, next_state, step_index)`

The adapter remains the only place allowed to know:

- die-to-grid mapping
- row snapping
- seed-completion semantics
- legality mask semantics

`engine/rl_env.py` should remain domain-agnostic.

## PPO Model Interface

Add a dedicated PPO model module rather than overloading `engine.gnn_proxy.py`.

Suggested file:

- `engine/ppo_policy.py`

Expected forward signature:

```python
def forward(
    graph_batch,
    canvas: torch.Tensor,
    current_macro_index: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

Outputs:

- policy logits over grid cells: `[batch, 1024]`
- value estimates: `[batch, 1]`

## Bring-Up Checklist

Each stage has a hard pass condition.

### Stage 1: Adapter/Env Integrity

Pass when:

- action mask excludes illegal cells
- sampled random actions always map to legal snapped coordinates
- seed-completed reward is defined for every step in an episode

### Stage 2: Masked Random Baseline

Pass when:

- `100` random episodes complete without invalid softmax or NaN policy issues
- aggregate episode return is stable enough to compute a mean and standard deviation

### Stage 3: Greedy Baseline

Pass when:

- greedy episode return beats masked random mean by at least `1 std`
- final analytic HPWL beats masked random mean final HPWL

### Stage 4: PPO Plumbing

Pass when:

- PPO runs for at least one rollout/update cycle without NaNs
- policy entropy, loss terms, and value estimates remain finite
- action masking never produces an all-invalid action vector

### Stage 5: PPO Beats Random

Pass when:

- PPO mean episode return exceeds masked random mean by at least `1 std`

### Stage 6: PPO Approaches Greedy

Pass when:

- PPO mean episode return reaches at least `90%` of greedy's mean return on the same benchmark and reward definition

Only after these six stages pass should broader PPO tuning begin.

## Open Questions Deferred

These are intentionally postponed:

- whether GraphSAGE is ultimately better than an edge-focused or hypergraph encoder
- whether orientation should become a second action head
- whether to move from `32 x 32` to `64 x 64`
- whether proxy and HPWL weights should be benchmark-specific
- whether PPO should later train across benchmarks

Those are v2 questions, not v1 bring-up questions.
