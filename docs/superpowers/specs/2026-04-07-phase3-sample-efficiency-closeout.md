# Phase 3 Sample-Efficiency Closeout

## Scope

This note closes the current `Phase 3A/3B` slice:

- benchmark ladder migration
- per-update sample-efficiency protocol
- teacher-strength sweep on the current beam-search teacher

It does **not** close the full Phase 3 research space. It closes the question:

> Does `behavior cloning -> PPO warm-start` provide a real early-training advantage over `PPO from scratch` on non-trivial chip-placement problems under a fixed update budget?

## Canonical Artifacts

- Benchmark ladder summary:
  - [summary.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_benchmark_ladder/summary.json)
- Teacher-strength sweep summary:
  - [summary.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_teacher_strength_sweep/summary.json)

## Main Result

The answer is **yes** on the current ladder.

Under a fixed `3`-update PPO budget and `5` random seeds:

- `tiny`
  - warm-start reaches beam level in `1.0` update on average
  - scratch reaches beam level in `2.2` updates on average
- `small`
  - `beam = 56.6`
  - `behavior_cloning = 56.6`
  - `ppo_warm_start = 56.6`
  - `ppo_scratch = 79.92`
  - warm-start reaches beam level in `1.0` update on average
  - scratch does not reach beam level within `3` updates
- `medium`
  - `beam = 72.5`
  - `behavior_cloning = 72.5`
  - `ppo_warm_start = 72.5`
  - `ppo_scratch = 102.6`
  - warm-start reaches beam level in `1.0` update on average
  - scratch does not reach beam level within `3` updates

The strongest version of the result is therefore:

> On the first non-trivial ladder benchmarks (`small` and `medium`), the warm-start advantage is primarily a **sample-efficiency** result rather than a final-asymptotic-quality result.

## Teacher-Strength Sweep

Teacher strength was varied by changing beam width while holding the rest of the protocol fixed.

### `small`

- `beam_width = 1, 2, 4` all produce the same teacher quality
- warm-start remains at the teacher level
- scratch remains behind under the same budget

Interpretation:

- the `small` benchmark is already hard enough to distinguish scratch from warm-start
- but it is not sensitive enough to teacher-width changes to serve as the main teacher-strength surface

### `medium`

- `beam_width = 1`
  - `beam = 78.5`
  - `warm_start = 78.5`
  - `scratch = 102.6`
  - scratch gap to beam `= 24.1`
- `beam_width = 2`
  - `beam = 72.5`
  - `warm_start = 72.5`
  - `scratch = 102.6`
  - scratch gap to beam `= 30.1`
- `beam_width = 4`
  - same as `beam_width = 2` on the current benchmark

Interpretation:

- teacher quality on `medium` improves from width `1` to `2`
- the warm-start policy continues to inherit the stronger teacher immediately
- scratch PPO does not improve under the same rollout/update budget
- on the current ladder, stronger teachers widen the fixed-budget warm-start vs scratch gap rather than erasing it

## What This Slice Proves

- The `search -> imitation -> PPO warm-start` path is no longer just an integration demo.
- Warm-start has a repeatable, non-trivial benefit on the current chip ladder.
- The meaningful metric for this phase is:
  - `updates_to_beam_level`
  - plus fixed-budget `gap_to_beam`

## What This Slice Does Not Prove

- It does **not** prove that warm-start improves the final asymptotic optimum after large PPO budgets.
- It does **not** prove that stronger search teachers will continue to help indefinitely beyond the current ladder.
- It does **not** justify adding new reward tricks, proxy variants, or curriculum changes yet.

## Current Recommendation

Do **not** add new RL modules before exhausting this comparison surface.

The next highest-value experiments are:

1. extend the per-update reporting into plot-ready artifacts
2. widen the teacher-strength sweep only where the teacher actually changes quality
3. keep the current reward and proxy setup fixed while doing so

## Deferred Work

The following remain explicitly deferred:

- wider search redesigns beyond simple beam-width control
- proxy-guided search as the main teacher
- reward reshaping or curriculum schedules
- revisiting the parked residual GNN proxy line from Phase 2

Those should only resume after the current sample-efficiency surface is fully characterized.
