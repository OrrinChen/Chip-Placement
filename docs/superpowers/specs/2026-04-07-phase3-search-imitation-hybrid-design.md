# Phase 3 Search And Imitation Hybrid Design

## Goal

Define how Phase 3 should incorporate heuristic/search and imitation learning without destabilizing the PPO bring-up path.

The design goal is not to replace PPO immediately. It is to introduce a stronger non-learned optimization baseline and a low-risk warm-start path for PPO, while preserving clean attribution when something improves or regresses.

## Decision

Phase 3 will use a three-stage structure:

1. keep PPO bring-up as the RL mainline
2. add search as a first-class optimization baseline, not just a smoke-test helper
3. add imitation as a warm-start path from search trajectories into PPO

This means the project will not treat pure PPO as the only serious optimization line.

## Why This Is The Right Split

The repo already learned a strong lesson in Phase 2: simple, well-conditioned models and diagnostics beat higher-capacity learned systems when data is still small or interfaces are still unstable.

That lesson transfers directly here.

If search is embedded too early inside PPO, then reward bugs, masking bugs, rollout bugs, and search bugs all become entangled. If imitation is added before search is stable, then the policy can inherit noise from a moving expert.

The clean decomposition is:

- search first, as a strong and independently measurable baseline
- imitation second, using frozen search trajectories
- PPO fine-tuning third, starting from the imitation-initialized policy

## Approaches Considered

### Approach A: Search Inside PPO From The Start

Use search directly in the PPO action-selection loop, such as beam-guided rollout expansion or search-assisted sampling.

Pros:

- potentially stronger policy improvement signal
- more direct hybrid story

Cons:

- highest debugging complexity
- runtime and credit-assignment become coupled
- difficult to tell whether gains come from PPO or search

Decision:

- rejected for Phase 3 v1

### Approach B: Parallel Search Baseline + Imitation Warm-Start + PPO Fine-Tune

Keep PPO as the RL mainline, but add a separate search line that uses the same state/action contract. Export trajectories from search and use them for behavior cloning. Initialize PPO policy weights from the cloned policy.

Pros:

- strongest attribution
- easiest to validate incrementally
- preserves a clean fallback if PPO underperforms
- lets search become a real competing optimization line rather than a hidden helper

Cons:

- more total components than pure PPO
- requires explicit trajectory format and serialization

Decision:

- recommended and chosen

### Approach C: Heuristic/Search Only, No RL

Treat the project as a search/proxy optimizer only and stop the PPO line.

Pros:

- likely strongest short-term optimization quality
- lowest RL bring-up risk

Cons:

- loses the RL narrative
- does not answer the Phase 3 roadmap question

Decision:

- rejected for now
- remains a fallback if PPO cannot beat simple baselines later

## Scope

This design covers:

- search baselines over the Phase 3 masked discrete action space
- trajectory generation for imitation learning
- behavior cloning warm-start into PPO
- evaluation framing for comparing random, greedy, search, BC, and PPO

This design does not cover:

- search inside PPO rollouts
- joint macro-order and position policies
- multi-design PPO training
- off-policy RL algorithms

## Search Baseline Design

### Shared Contract

Search must use the same:

- fixed macro order `(-area, name)`
- `32 x 32` masked discrete grid
- adapter-generated legality mask
- chip reward-facing state contract already used by PPO bring-up

This is important because search is intended to be an apples-to-apples comparator, not a separate problem formulation.

### Search Variants

Phase 3 hybrid v1 will implement two search baselines:

1. `greedy_one_step`
2. `beam_search`

`greedy_one_step` already exists in minimal form and remains the cheap ceiling for bring-up.

`beam_search` is the new first-class search baseline.

### Beam Search

Beam search expands the top `k` partial placements at each step under the current fixed macro order.

Each candidate expansion:

- enumerates valid actions from the current mask
- scores them using analytic HPWL for v1
- retains the top `beam_width` partial placements

The first version uses analytic HPWL scoring instead of proxy scoring so that:

- it remains independent from the PPO reward implementation
- it provides a cleaner ceiling
- it avoids entangling search quality with proxy calibration

Later versions may add proxy-guided search, but not in the first hybrid slice.

## Imitation Learning Design

### Expert Source

The initial expert policy is search-derived, not human-labeled.

Trajectory priority:

1. beam search
2. greedy one-step

Random rollouts are not demonstration data.

### Demonstration Format

Each trajectory record should store:

- design identifier
- rollout type
- step index
- current macro name
- observation tensors or a reproducible observation payload
- action mask
- chosen action index
- final episode score metadata

The format must be serializable so the same demonstrations can be reused across experiments without rerunning search.

### Behavior Cloning Objective

Behavior cloning trains the policy head to imitate the expert action distribution.

The first version uses:

- masked cross-entropy over valid grid actions
- the same observation encoder stack as PPO
- no value loss

The BC model is therefore not a separate architecture. It is the PPO policy architecture trained with a supervised objective.

## PPO Warm-Start Design

### What Transfers

Warm-start transfers:

- graph encoder weights
- canvas encoder weights
- FiLM conditioning weights
- policy head weights

Warm-start does not transfer:

- value head weights

The value head is initialized fresh for PPO.

### Why Value Head Stays Fresh

The BC phase teaches action preference, not a trustworthy value function. Reusing a BC-trained value head would introduce noisy bootstrap targets into PPO from the start.

## Reward Relationship

The PPO reward contract remains the one already fixed in the PPO model spec:

- seed-completed frozen linear proxy delta
- analytic HPWL delta
- legality penalty

Search and imitation do not redefine reward.

Important separation:

- search scoring in v1 uses analytic HPWL
- PPO reward continues using proxy + analytic shaping

This asymmetry is intentional. It keeps the search baseline from becoming a tautological copy of the PPO reward path.

## Component Boundaries

### Search

- file family: `chip/experiments/phase3_search.py`
- responsibility: run deterministic search baselines and emit comparable metrics and optional trajectories

### Demonstration IO

- file family: `chip/imitation.py`
- responsibility: trajectory schema, serialization, loading, batching helpers

### Behavior Cloning

- file family: `chip/experiments/phase3_imitation.py`
- responsibility: train supervised policy from search trajectories and export warm-start checkpoints

### PPO Warm-Start

- file family: `chip/experiments/phase3_ppo.py`
- responsibility: load optional BC checkpoint before PPO training

## Evaluation Order

Bring-up should follow this order:

1. masked random baseline
2. greedy one-step baseline
3. beam search baseline
4. behavior cloning from search trajectories
5. PPO from scratch
6. PPO warm-started from BC

This ordering matters because each stage establishes a new ceiling or floor for the next one.

## Bring-Up Gates

### Search Gate

Beam search must:

- respect action masks with zero invalid actions
- match or beat greedy one-step final analytic HPWL on the same tiny benchmark

### Imitation Gate

Behavior cloning must:

- recover the search action on held-out tiny-benchmark states with high masked-action accuracy
- produce rollouts that beat masked random and approach greedy/beam behavior

### PPO Warm-Start Gate

Warm-started PPO must:

- train without destabilizing masking or reward code
- beat masked random faster than PPO-from-scratch
- not underperform the frozen BC policy at the first evaluation checkpoint

## Risks

### Search Becomes The Best Method

This is acceptable. The project should explicitly allow search to become the strongest optimization line while PPO remains the strongest RL line.

### Demonstration Quality Is Too Narrow

If search trajectories collapse to a tiny behavioral manifold, BC may overfit and PPO fine-tuning may add little value. This is acceptable in v1. The immediate goal is warm-start stability, not expert diversity.

### PPO Learns Nothing Beyond BC

This is also acceptable in v1. In that case, the correct story is that imitation provides a stable deployment-quality policy while PPO remains exploratory.

## Success Criteria

Phase 3 hybrid v1 is successful if:

- beam search is implemented and beats or matches greedy on the shared action space
- search trajectories can be serialized and replayed
- behavior cloning can train on those trajectories and run inference with the PPO policy architecture
- PPO can optionally initialize from the BC checkpoint without changing the environment contract

## Decision Summary

Phase 3 will not treat heuristic/search as a hidden helper inside PPO.

Instead it will:

- elevate search to a real optimization baseline
- use imitation to bridge search and PPO
- keep PPO as the RL mainline without making it the only optimization story
