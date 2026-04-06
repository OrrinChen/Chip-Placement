# Task Memory

## 2026-04-05

### Current State

- `Phase 1` roadmap acceptance is merged on `main` at commit `93693b1`.
- The stable acceptance benchmark set is:
  - `mgc_pci_bridge32_1`
  - `mgc_pci_bridge32_2`
  - `mgc_des_perf_2`
- The benchmark manifest is stored at [experiments/chip/phase1_acceptance_benchmark_manifest.json](C:/Users/14574/Chip/experiments/chip/phase1_acceptance_benchmark_manifest.json).
- The acceptance results table is stored at [experiments/chip/phase1_acceptance_results.json](C:/Users/14574/Chip/experiments/chip/phase1_acceptance_results.json).
- `OpenROAD` execution is Docker-based and GPU-first, with CPU fallback when GPU runtime support is unavailable.

### Acceptance Status

- The Phase 1 acceptance flow now covers:
  - `3` public ISPD benchmarks
  - `SA` placement injection through the OpenROAD Tcl path
  - one shared OpenROAD tail flow for `built_in` and `sa_injected`
  - final metrics for `HPWL`, `total routed wirelength`, `WNS`, `TNS`, `DRC violation count`, and runtime
- The acceptance results now carry an explicit `phase1_gate_status` field so the infrastructure pass and optimization-quality gap are not conflated.
- Fresh verification after merge:
  - `pytest tests/chip -v`
  - `43 passed in 1.13s`

### Known Limits

- `mgc_fft` was excluded from the stable set because the built-in OpenROAD run stalled in the `global_route` tail for too long.
- The current scale spread is acceptable for Phase 1, but still skewed:
  - `mgc_pci_bridge32_1` and `mgc_pci_bridge32_2` are close variants
  - the manifest includes a `scale_note` calling this out
- The results satisfy the acceptance chain, but `sa_is_valid` is still `false` on the stable set.
  - The stable rows are marked `phase1_gate_status = infra_pass / quality_pending`
  - The routed and timing metrics are informative only and are not a strict apples-to-apples comparison against the built-in placer until SA legality is fixed

### Next Good Entry Points

- Improve SA legality and post-injection quality before using these designs as stronger baselines.
- Start `Phase 2` proxy work using the stable benchmark manifest and results artifacts.
- If better scale diversity is needed later, replace or extend one of the `pci_bridge32` variants with a stable medium-large public ISPD design.

## 2026-04-05 Phase 2 Foundation

### Current Branch State

- Active worktree: [phase2-foundation](C:/Users/14574/Chip/.worktrees/phase2-foundation)
- Active branch: `codex/phase2-foundation`
- Foundation checkpoint commit: `56bed9d`

### Phase 2 Foundation Progress

- `engine/contracts.py` now exposes the neutral protocol set centered on:
  - `DecisionEnv`
  - `ConstraintChecker`
  - `Evaluator`
  - `SurrogateModel`
  - `SearchStrategy`
  - `TrainingStrategy`
- `TransitionModel` remains as a compatibility alias for now, so old references do not break while Phase 2 code starts using the new API.
- `chip/legalization.py` now provides a lightweight OpenROAD legalization path that injects placement, runs `detailed_placement`, and writes out a legalized DEF without paying the full routing tail cost.

### Benchmark Assets

- The worktree now has the official public benchmark assets staged under `tmp/`:
  - ISPD 2014 Benchmark Suite A
  - ISPD 2012 benchmark package for `contest.lib` and official `sdc` files
- These assets are local worktree inputs only and are not intended to be committed.

### Phase 2 Dataset Status

- The first Phase 2 sampling manifest has been generated at [phase2_dataset_manifest.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_manifest.json).
- The current manifest contains `21` planned samples:
  - `4` random perturbations and `3` SA-truncated samples for each of the `3` stable benchmarks
- Current manifest rows already carry `phase1_gate_status`, so downstream proxy work can keep infrastructure readiness and optimization-quality caveats separate.

### Background Validation

- A fresh `phase1_acceptance` rerun is currently running in the Phase 2 worktree against the downloaded official assets.
- Purpose:
  - verify whether the new legalization pass changes `sa_is_valid`
  - confirm the stable benchmark flow still works with locally reconstructed official inputs
