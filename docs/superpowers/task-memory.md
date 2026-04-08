# Task Memory

## 2026-04-06 Phase 2 Stratified Rerun

### Current Branch State

- Active worktree: [phase2-foundation](C:/Users/14574/Chip/.worktrees/phase2-foundation)
- Active branch: `codex/phase2-foundation`
- Most recent commits on this branch:
  - `91a03c9` `feat: add stratified phase2 perturbation sampling`
  - `13c7f46` `fix: harden phase2 sample evaluation`
- Fresh verification after the stratified sampling changes:
  - `pytest tests/chip -q`
  - `97 passed in 3.56s`

### Perturbation Repair Outcome

- Timeout cleanup for Docker-backed OpenROAD evaluation is now explicit.
  - `docker run` containers are named deterministically.
  - `subprocess.TimeoutExpired` now triggers `docker rm -f <container>` cleanup in [openroad_bridge.py](C:/Users/14574/Chip/.worktrees/phase2-foundation/chip/openroad_bridge.py).
  - A real timeout smoke check left `docker ps` empty afterward, so the old container leak is fixed.
- Random perturbation now records displacement diagnostics in each successful dataset row:
  - `displacement_median_sites`
  - `displacement_p90_sites`
  - `displacement_max_sites`
  - `moved_fraction`
- Sweep results are recorded under [phase2_repair_sweeps](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_repair_sweeps) with the summary in [phase2_perturbation_sweep_summary.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_repair_sweeps/phase2_perturbation_sweep_summary.json).

### Stratified Sampling Decision

- The current `full` batch defaults to stratified random perturbation instead of a single random tier.
- Current default tiers:
  - primary band: `70%` of random samples at `move_probability = 0.10`, `max_x_fraction = 0.10`, `row_window = 1`
  - hard negative: `30%` of random samples at `move_probability = 0.20`, `max_x_fraction = 0.10`, `row_window = 1`
- Supporting evidence from the sweep summary:
  - `move_probability = 0.10` on `mgc_pci_bridge32_1` landed near `1.325x` seed HPWL at about `5.03s`
  - `move_probability = 0.20` on `mgc_pci_bridge32_1` landed near `1.600x` seed HPWL at about `5.15s`
- Intent:
  - keep most training mass near the useful `~1.3x` band
  - extend the negative tail so the proxy sees placements well below the seed quality band

### Full Stratified Collector Status

- A full rerun is currently in flight:
  - command:
    - `python -m chip.experiments.phase2_dataset --collect --batch full --random-samples 100 --sa-truncation-steps --benchmarks mgc_pci_bridge32_1 mgc_pci_bridge32_2 mgc_des_perf_2 --benchmark-root tmp --output experiments/chip/phase2_dataset_full_stratified.json`
  - output:
    - [phase2_dataset_full_stratified.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_full_stratified.json)
  - logs:
    - [phase2_full_stratified.out](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_full_stratified.out)
    - [phase2_full_stratified.err](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_full_stratified.err)
  - background PID at the current checkpoint:
    - `14888`
- Current checkpoint at time of writing:
  - `275` rows written
  - `270 success`
  - `5 timeout`
  - benchmark counts:
    - `mgc_pci_bridge32_1 = 100`
    - `mgc_pci_bridge32_2 = 100`
    - `mgc_des_perf_2 = 75`

### Timeout Diagnosis for `mgc_des_perf_2`

- The timeouts are benchmark-specific and config-specific, not flow-wide.
- Evidence:
  - `mgc_pci_bridge32_1` and `mgc_pci_bridge32_2` completed `100/100` successfully under the same stratified collector.
  - `mgc_des_perf_2` started timing out exactly when the random sample plan crossed from the `0.10` tier to the `0.20` tier.
  - Mapping from the stratified plan:
    - `rand_0069` is still `move_probability = 0.10`
    - `rand_0070` onward are `move_probability = 0.20`
  - Quantitative comparison:
    - `rand_0069`
      - `hpwl_ratio_vs_seed ≈ 1.494`
      - `displacement_p90_sites ≈ 35`
      - OpenROAD runtime `~21.7s`
    - `rand_0070`
      - `hpwl_ratio_vs_seed ≈ 1.892`
      - `displacement_p90_sites ≈ 122`
      - `60s` diagnostic rerun still timed out
- OpenROAD stage evidence for `rand_0070`:
  - `detailed_placement` finishes
  - `legalized HPWL == original HPWL`
  - the run then stalls after entering `global_route`
- Interpretation:
  - legality is not the problem
  - legalization is not overwriting the placement
  - the `0.20` hard-negative tier pushes `mgc_des_perf_2` placements into a router slow zone that the two `pci_bridge32` designs tolerate but `des_perf_2` does not

### Recommended Next Entry Point

- Treat the current issue as benchmark-specific stratification instability.
- Keep the current `0.10/0.20` `70/30` split for `mgc_pci_bridge32_1` and `mgc_pci_bridge32_2`.
- For `mgc_des_perf_2`, the next most targeted experiment is:
  - lower the hard-negative tier from `0.20` to `0.15`, or
  - use a benchmark-specific split such as `0.10/0.15`
- Do not change the OpenROAD flow before changing the `mgc_des_perf_2` perturbation tier.

## 2026-04-06 Phase 2 Overnight

### Current Branch State

- Active worktree: [phase2-foundation](C:/Users/14574/Chip/.worktrees/phase2-foundation)
- Active branch: `codex/phase2-foundation`
- Fresh verification after the latest Phase 2 collector and proxy changes:
  - `pytest tests/engine tests/chip -v`
  - `82 passed in 4.47s`

### Phase 2 Data Pipeline Status

- `chip/experiments/phase2_dataset.py` now uses a fully placed OpenROAD seed DEF instead of the `floorplan_clean.def` base for perturb / legalize / evaluate.
  - `floorplan_clean.def` leaves most standard cells `UNPLACED`, so incremental injection on that base was not a valid legalization strategy.
  - The collector now ensures `seed_placed.def` exists and reuses it as the mutable placement base.
- Random sample generation now preserves legality before OpenROAD sees it.
  - Random perturbation retries neighbors until it finds a legal move or stays put.
  - SA-truncated samples fall back to the initial legal seed when the truncated SA state is still illegal.
- OpenROAD evaluation now scores the `legalized.def` directly instead of reinjecting every cell back into a clean DEF.
- Collector persistence is now resumable.
  - Existing rows in the output JSON are skipped on restart.
  - New rows are written back to the output JSON after each sample, not only at the end of the batch.

### Smoke Results

- Stable single-sample smoke:
  - Command:
    - `python -m chip.experiments.phase2_dataset --collect --benchmarks mgc_pci_bridge32_1 --random-samples 1 --sa-truncation-steps --timeout-seconds 600 --output experiments/chip/phase2_dataset_smoke.json`
  - Result:
    - `rand_0000` succeeded
    - OpenROAD runtime: `4.798377600003732s`
    - Output: [phase2_dataset_smoke.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_smoke.json)
- Cross-benchmark smoke:
  - Command:
    - `python -m chip.experiments.phase2_dataset --collect --benchmarks mgc_pci_bridge32_2 mgc_des_perf_2 --random-samples 1 --sa-truncation-steps --timeout-seconds 900 --output experiments/chip/phase2_dataset_cross_benchmark_smoke.json`
  - Results:
    - `mgc_pci_bridge32_2/rand_0000`: success, OpenROAD runtime `4.742476200000965s`
    - `mgc_des_perf_2/rand_0000`: success, OpenROAD runtime `18.05286370000249s`
    - Output: [phase2_dataset_cross_benchmark_smoke.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_cross_benchmark_smoke.json)

### Overnight Runs In Flight

- Full Phase 2 collector is running in the background with incremental persistence:
  - Command:
    - `python -m chip.experiments.phase2_dataset --collect --benchmarks mgc_pci_bridge32_1 mgc_pci_bridge32_2 mgc_des_perf_2 --batch full --random-samples 160 --sa-truncation-steps 1 5 10 25 50 100 250 500 --timeout-seconds 900 --output experiments/chip/phase2_dataset_full.json`
  - Output JSON:
    - [phase2_dataset_full.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_full.json)
  - Sample artifacts:
    - [phase2_samples](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_samples)
  - Current background collector PID at time of writing:
    - `2072`
  - Current collector logs:
    - [phase2_full.out](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_full.out)
    - [phase2_full.err](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_full.err)
  - Mid-run checkpoints:
    - `2026-04-06 08:20 America/Toronto`: `449` successful rows
      - `mgc_pci_bridge32_1`: `168`
      - `mgc_pci_bridge32_2`: `168`
      - `mgc_des_perf_2`: `113`
    - `2026-04-06 08:23 America/Toronto`: `474` successful rows, so the collector was still advancing through `mgc_des_perf_2`
- Automatic proxy follow-up is also queued in the background.
  - It waits for the collector process to exit, then runs:
    - `python -m chip.experiments.phase2_proxy --dataset-results experiments/chip/phase2_dataset_full.json --benchmark-root tmp --output-dir experiments/chip/phase2_proxy_full --epochs 300`
  - Planned artifacts:
    - [phase2_proxy_full](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_full)
  - Follow-up logs:
    - [phase2_proxy_full.out](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_proxy_full.out)
    - [phase2_proxy_full.err](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/phase2_proxy_full.err)

### Proxy Training Status

- `chip/experiments/phase2_proxy.py` now supports leave-one-benchmark-out cross-design validation.
  - The report now includes `cross_design_folds`, `cross_design_mean_spearman`, and `cross_design_min_spearman`.
  - A final model checkpoint is still trained on the full collected dataset after the validation folds complete.
- Proxy training is now bounded so it can finish on real benchmark graphs.
  - The loader downsamples successful rows per benchmark before parsing graphs.
  - Training and inference run in graph mini-batches instead of one giant full-batch tensor.
  - Current defaults:
    - `max_rows_per_benchmark = 8`
    - `batch_size = 2`
- The design-split contract is covered by unit tests in [test_phase2_proxy.py](C:/Users/14574/Chip/.worktrees/phase2-foundation/tests/chip/test_phase2_proxy.py).
- A snapshot proxy run completed successfully before the full collector finished.
  - Command:
    - `python -m chip.experiments.phase2_proxy --dataset-results experiments/chip/phase2_dataset_full.json --benchmark-root tmp --output-dir experiments/chip/phase2_proxy_snapshot --epochs 120 --max-rows-per-benchmark 8 --batch-size 2`
  - Artifacts:
    - [phase2_proxy_snapshot](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_snapshot)
  - Snapshot report highlights:
    - `cross_design_mean_spearman = 0.08773058773058771`
    - `cross_design_min_spearman = -0.14761904761904765`
    - `full_training_row_count = 24`
    - Holdout fold mean Spearman:
      - `mgc_des_perf_2`: `-0.14761904761904765`
      - `mgc_pci_bridge32_1`: `0.043243243243243246`
      - `mgc_pci_bridge32_2`: `0.36756756756756753`

### Known Caveats

- `cvxpy` still emits `ortools 9.15` compatibility warnings on import.
  - These are warnings only; they have not blocked the dataset collector or proxy tests.
- The current overnight collector processes benchmarks sequentially.
  - Early incremental rows will appear in `phase2_dataset_full.json` from `mgc_pci_bridge32_1` first, then the later benchmarks.
- The current proxy follow-up waits for the collector PID recorded above.
  - If the collector is manually restarted under a new PID, the follow-up command should also be relaunched so it waits on the new process.
- The current snapshot proxy report is real but still weak.
  - Cross-design ranking signal is low on the bounded `24`-row training subset, so Phase 2 is not yet a strong quality win even though the pipeline is end-to-end functional.

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
- The current manifest is explicitly tagged as `batch = smoke`.
- Current manifest rows already carry `phase1_gate_status`, so downstream proxy work can keep infrastructure readiness and optimization-quality caveats separate.
- Dataset collection scaffolding now distinguishes `success`, `timeout`, and `error` outcomes so large OpenROAD batches can retain failure metadata instead of silently dropping hard cases.

### Background Validation

- A fresh `phase1_acceptance` rerun is currently running in the Phase 2 worktree against the downloaded official assets.
- Purpose:
  - verify whether the new legalization pass changes `sa_is_valid`
  - confirm the stable benchmark flow still works with locally reconstructed official inputs
- A direct legality spot-check on [legalized.def](C:/Users/14574/Chip/.worktrees/phase2-foundation/tmp/mgc_pci_bridge32_1/legalized.def) now passes after adding an epsilon tolerance to the chip legality checker.
  - The prior `quality_pending` state was partly due to `1e-14` scale floating-point residue in the checker rather than a real placement defect

## 2026-04-07 Phase 2 Proxy Diagnosis

### Dataset Status

- The Phase 2 production dataset is now finalized at [phase2_dataset_full_stratified_main.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_full_stratified_main.json).
- The canonical summary is stored at [phase2_dataset_full_stratified_main_summary.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_dataset_full_stratified_main_summary.json).
- Final dataset status:
  - `300/300 success`
  - `0 timeout`
  - `0 error`
- The shared OpenROAD tail flow remains valid.
- `mgc_des_perf_2` now uses a benchmark-specific hard-tier cap of `0.15`.
  - `mgc_pci_bridge32_1` and `mgc_pci_bridge32_2` retain `0.20`.
  - This preserves stratified difficulty while avoiding the router stability cliff on `mgc_des_perf_2`.

### Proxy Gate Status

- The staged proxy rerun was intentionally stopped after the most informative first fold:
  - `same-design / mgc_pci_bridge32_1`
- The original GNN-only first fold report is stored at [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_main_same_design_pci1/phase2_proxy_report.json).
- That fold remained negatively correlated:
  - `mean_spearman = -0.09624060150375939`
  - `hpwl = -0.11278195488721804`
  - `total_routed_wirelength = -0.07969924812030074`
- This was enough to trigger an early stop.
  - The dataset was already stable, so the blocker moved cleanly to proxy learning rather than dataset generation or flow stability.

### Proxy Diagnosis

- Train-vs-validation sanity on the original GNN fold showed:
  - `train mean Spearman = +0.0865`
  - `validation mean Spearman = -0.0737`
- This ruled out a trivial target-direction/sign bug as the primary cause.
- Row-level geometry already carries strong signal on `mgc_pci_bridge32_1`:
  - `moved_fraction` vs `hpwl`: `0.879`
  - `displacement_p90_sites` vs `hpwl`: `0.888`
  - `moved_fraction` vs `total_routed_wirelength`: `0.869`
  - `displacement_p90_sites` vs `total_routed_wirelength`: `0.890`
- Current blocker is therefore:
  - proxy representation failure
  - not dataset instability
  - not shared flow failure
  - not a trivial target-orientation bug

### Minimal Fix Loop

- `chip/experiments/phase2_proxy.py` now supports:
  - `--model-family gnn|graph-stats`
  - graph-level geometry features derived from:
    - `moved_fraction`
    - `displacement_median_sites`
    - `displacement_p90_sites`
    - `displacement_max_sites`
  - fold reports now include `train_per_target_spearman` and `train_mean_spearman`
  - JSON loading now uses `utf-8-sig` so BOM-encoded dataset files do not break the proxy entrypoint
- `engine/gnn_proxy.py` now supports graph-level features in batching and readout.

### Real Fold Re-checks

- The cheap non-GNN sanity probe is stored at [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_graph_stats_pci1/phase2_proxy_report.json).
  - `model_family = graph-stats`
  - `same-design mean_spearman = 0.925563909774436`
  - `train_mean_spearman = 0.9311767463666198`
- The minimally enhanced GNN re-check is stored at [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_graphfeat_pci1/phase2_proxy_report.json).
  - `model_family = gnn`
  - `same-design mean_spearman = 0.4496240601503759`
  - `train_mean_spearman = 0.7307196436943273`
- Interpretation:
  - the task is learnable on the finalized dataset
  - graph-level seed-relative geometry is sufficient to pass the same-design sanity fold
  - the current GNN still underfits relative to that cheap baseline, even after adding graph-level features

### Current Entry Point

- Do not continue blind fold reruns with the old proxy configuration.
- The next highest-value step is to keep improving the GNN representation, starting with more explicit seed-relative node features such as:
  - `dx`
  - `dy`
  - moved flag
  - local displacement magnitude
- Phase 2 is no longer blocked by dataset construction.
  - It is now blocked by the proxy model family not yet matching the information available in simple geometry-based baselines.

## 2026-04-07 Seed-Relative Node Features

### What Changed

- `chip/experiments/phase2_proxy.py` now augments each sample graph with seed-relative node features computed against the benchmark-local `seed_placed.def` reference:
  - `dx`
  - `dy`
  - `dx_abs`
  - `dy_abs`
  - `dr`
  - `moved_flag`
- The reference is resolved per benchmark from the existing `tmp/<benchmark>/seed_placed.def` artifact.
- These features are normalized by die width and die height so they are scale-aware within each benchmark.

### Verification

- Targeted tests remain green:
  - `pytest tests/chip/test_phase2_proxy.py -q`
  - `18 passed`
  - `pytest tests/engine/test_gnn_proxy.py -q`
  - `5 passed`

### Same-Design Re-check

- The updated GNN same-design rerun for `mgc_pci_bridge32_1` is stored at [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_seedrel_pci1/phase2_proxy_report.json).
- Results:
  - `same-design mean_spearman = 0.5721804511278195`
  - `hpwl = 0.5593984962406015`
  - `total_routed_wirelength = 0.5849624060150376`
  - `train_mean_spearman = 0.7845171120487577`
- Interpretation:
  - explicit seed-relative node features materially improved the GNN
  - previous graph-feature-only GNN result was `0.4496240601503759`
  - current node-augmented GNN is still below the `0.8` same-design gate

### Current Read

- The same-design fold is now clearly learnable by geometry-aware models:
  - `graph-stats` baseline remains at `0.925563909774436`
  - node-augmented GNN improved to `0.5721804511278195`
- This narrows the remaining blocker further:
  - the missing ingredient is no longer just seed-relative node information
  - the remaining gap is likely in pooling/readout or in how message passing dilutes the strongest displacement signal

## 2026-04-07 Pooling Sanity Check

### Step A Results on `same-design / mgc_pci_bridge32_1`

- Mean pooling reference:
  - artifact: [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_seedrel_pci1/phase2_proxy_report.json)
  - `same-design mean_spearman = 0.5721804511278195`
  - `train_mean_spearman = 0.7845171120487577`
- Sum pooling:
  - artifact: [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_sum_pci1/phase2_proxy_report.json)
  - `same-design mean_spearman = 0.45263157894736833`
  - `train_mean_spearman = 0.6966596343178623`
- `concat(mean, max, std)` pooling:
  - artifact: [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_meanmaxstd_pci1/phase2_proxy_report.json)
  - `same-design mean_spearman = 0.04962406015037593`
  - `train_mean_spearman = 0.3817393342709799`

### Interpretation

- Step A did not produce a readout-only fix.
- Mean pooling remains the strongest of the three tested options on this fold.
- Sum pooling is directionally positive but worse than mean.
- `concat(mean, max, std)` significantly degrades both train and validation ranking.
- Current evidence does not support the hypothesis that mean pooling alone is the dominant bottleneck.
- The remaining blocker is more likely in the interaction between message passing and displacement-heavy node signals than in the final readout choice by itself.

### Next Best Diagnostic

- The next narrow diagnostic should be the plumbing check proposed after Step A:
  - freeze or bypass the GNN branch
  - route graph features and a minimal node-readout path through the head
  - verify whether the model can recover the `graph-stats` baseline ceiling inside the current proxy plumbing

## 2026-04-07 Step B Plumbing Check

### Graph-Features-Only Check Inside Current GNN Plumbing

- Artifact: [phase2_proxy_report.json](C:/Users/14574/Chip/.worktrees/phase2-foundation/experiments/chip/phase2_proxy_gnn_graphfeatures_only_pci1/phase2_proxy_report.json)
- Configuration:
  - `model_family = gnn`
  - `pooling_mode = mean`
  - `node_path_scale = 0.0`
  - graph features still attached through the normal proxy path
- Result on `same-design / mgc_pci_bridge32_1`:
  - `same-design mean_spearman = -0.7556390977443608`
  - `train_mean_spearman = -0.8455227379278014`

### Interpretation

- This is worse than the node-augmented GNN and much worse than the external `graph-stats` baseline.
- The result does not support a simple "GNN branch is negative, graph_features branch alone already solves it inside the current head" story.
- A quick in-memory weight check on the trained fold showed:
  - first-layer node-branch weight norm: `~4.19`
  - first-layer graph-feature weight norm: `~158.95`
- This makes a dead concat or obviously ignored graph-feature path less likely.
- More plausible current read:
  - graph features do reach the head
  - but the current MLP head and optimization regime can fit them in the wrong direction on this tiny fold

### Current Conclusion

- Step B did not reveal a simple plumbing disconnect.
- The remaining issue is now narrower:
  - the current proxy architecture does not preserve the strong geometry baseline when graph features are routed through the learned GNN head
  - this points toward head parameterization or residual-design work rather than more dataset debugging

## 2026-04-07 Graph-Features Sign Follow-up

### What Was Checked

- Target / feature preprocessing / loss wiring was compared directly between:
  - `train_graph_stats_model`
  - `train_proxy_model`
- Findings:
  - graph-feature preprocessing is shared
    - both paths use `build_graph_level_feature_matrix`
    - both paths use `_normalize_feature_matrix`
  - there is no explicit sign flip in either path
  - the main difference is:
    - `graph-stats` uses closed-form linear least squares on raw targets
    - the learned proxy path uses `Adam + ReLU MLP` with MSE against z-scored targets

### Isolation Experiment

- A standalone graph-feature MLP was trained outside the GNN plumbing using:
  - the same standardized graph features
  - the same target normalization style as `train_proxy_model`
  - the same `Adam`-style regression setup
- Result on `same-design / mgc_pci_bridge32_1`:
  - train Spearman stayed negative
  - validation Spearman stayed negative
- In contrast, the closed-form linear baseline on the same normalized graph features remained strongly positive:
  - train `hpwl` Spearman `= 0.9358`
  - validation `hpwl` Spearman `= 0.9278`
  - train `total_routed_wirelength` Spearman `= 0.9265`
  - validation `total_routed_wirelength` Spearman `= 0.9233`

### Interpretation

- The negative `graph_features-only` result is not explained by:
  - a dead graph-feature concat
  - a missing normalization step
  - a trivial target-direction mismatch
- More precise read:
  - on this tiny fold, the current learned MLP-style head can learn the graph-feature signal in the wrong direction
  - the closed-form linear baseline does not have this failure mode
- This shifts the next architecture decision:
  - if a graph-feature baseline is used as the stable signal source, it should likely be kept as a frozen linear/base predictor rather than routed through the current learned MLP head and allowed to drift

## 2026-04-07 Capacity vs Regime Sanity Checks

### Setup

- Same fold: `same-design / mgc_pci_bridge32_1`
- Same graph-level feature matrix:
  - standardized with `_normalize_feature_matrix`
- Same target setup for learned models:
  - raw targets for reporting
  - z-scored targets inside the optimization loss
  - `Adam`
  - `MSE`
  - mini-batch size `2`

### Results

- `graph_feature_mlp`
  - hidden size `64`
  - epochs `200`
  - train `hpwl` Spearman `= -0.8451`
  - validation `hpwl` Spearman `= -0.7534`
  - train `total_routed_wirelength` Spearman `= -0.8446`
  - validation `total_routed_wirelength` Spearman `= -0.7444`
- `graph_feature_linear_adam`
  - no hidden layer
  - epochs `200`
  - train `hpwl` Spearman `= 0.8673`
  - validation `hpwl` Spearman `= 0.8316`
  - train `total_routed_wirelength` Spearman `= 0.8319`
  - validation `total_routed_wirelength` Spearman `= 0.7850`
- `graph_feature_linear_adam`
  - no hidden layer
  - epochs `2000`
  - train `hpwl` Spearman `= 0.8475`
  - validation `hpwl` Spearman `= 0.8045`
  - train `total_routed_wirelength` Spearman `= 0.8426`
  - validation `total_routed_wirelength` Spearman `= 0.8015`

### Interpretation

- The learned-head failure is not caused purely by:
  - target normalization
  - `Adam`
  - `MSE`
- A linear model trained under the same learned optimization regime still recovers a strong positive same-design signal.
- The failure mode is therefore more specific:
  - large-capacity `ReLU MLP` heads on this tiny fold can learn the graph-feature signal in the wrong direction
  - a linear base model does not exhibit that sign-flip behavior

### Architecture Implication

- This supports a more specific next move than a generic residual:
  - if a stable geometry branch is used, it should be a frozen linear/base predictor
  - the learned GNN branch should be added as a constrained residual on top of that base, rather than being allowed to relearn the base signal through another flexible MLP head

## 2026-04-07 Frozen Linear Base Gate Pass

### Implementation

- Added `frozen-linear-base` to `chip/experiments/phase2_proxy.py`.
- The new path is fold-aware:
  - base model is fit only on the current fold's train split
  - validation predictions come from that fold-local base
- Added fixed `residual_alpha` plumbing:
  - `alpha = 0` skips graph loading and uses the frozen linear base only
  - `alpha > 0` trains the GNN on residual targets `y - y_base`
- Added tests for:
  - fold-aware base predictions differing from full-data fit
  - `alpha = 0` base-only path skipping graph loading

### Verification

- `pytest tests/chip/test_phase2_proxy.py -q` -> `20 passed`
- `pytest tests/engine/test_gnn_proxy.py -q` -> `8 passed`

### Real-Fold Results

- `same-design / mgc_pci_bridge32_1`, `alpha = 0.0`
  - report: `experiments/chip/phase2_proxy_frozen_base_alpha0_pci1/phase2_proxy_report.json`
  - validation mean Spearman `= 0.9256`
  - train mean Spearman `= 0.9312`
  - this reproduces the strong graph-stats baseline and clears the `same-design > 0.8` gate

### Full Phase 2 Proxy Result (Current Production Proxy)

- ran full fold set with `model_family = frozen-linear-base`, `residual_alpha = 0.0`
  - report: `experiments/chip/phase2_proxy_frozen_base_alpha0_full/phase2_proxy_report.json`
- aggregate metrics:
  - same-design mean Spearman `= 0.9008`
  - same-design min Spearman `= 0.8865`
  - cross-design mean Spearman `= 0.3859`
  - cross-design min Spearman `= 0.2076`
- gate status:
  - `same_design_status = pass`
  - `cross_design_status = pass`
  - `cross_design_stretch_status = fail`
  - `phase3_foundation_ready = true`

### Decision

- Phase 2 is no longer blocked by dataset instability or proxy sign issues.
- At the current fold size, the production proxy should be the frozen linear base (`alpha = 0`) rather than the learned GNN residual.
- Small-`alpha` residual sweeps are still open as follow-up work, but they are no longer on the critical path for the Phase 2 gate.

## 2026-04-07 Phase 2 Closeout Framing

### Cross-Design Limitation

- The current production proxy closes the same-design gate strongly:
  - same-design mean Spearman `= 0.9008`
- Cross-design generalization remains materially weaker:
  - cross-design mean Spearman `= 0.3859`
  - cross-design min Spearman `= 0.2076`
- This is an explicit Phase 2 closeout limitation, not a bug:
  - the current geometry-heavy feature set generalizes well within a benchmark's perturbation family
  - it does not yet transfer strongly across different designs
- Carry this forward into Phase 3 as a known limitation of the current proxy/data representation, not as a reopened Phase 2 dataset or flow issue.

### Residual GNN Status

- The `alpha > 0` residual GNN branch is parked, not pending.
- Reason:
  - frozen linear base already passes the approved Phase 2 gate cleanly
  - the learned residual branch is not on the critical path
  - small-data GNN experiments have repeatedly shown unstable or low-value behavior relative to the linear base
- Restart condition:
  - do not revisit the residual GNN until the proxy training set grows to at least `256` samples per benchmark in the production dataset
  - before that threshold, the production proxy remains the frozen linear base with `alpha = 0.0`

### Closeout Checklist

- `phase2_gate` is code-derived, not manually filled:
  - see `build_phase2_gate()` in `chip/experiments/phase2_proxy.py`
- Production default is aligned with the closeout result:
  - `DEFAULT_FROZEN_BASE_ALPHA = 0.0`
- Fold-aware safeguards are explicitly tested:
  - fold-local base differs from full-data fit
  - fold A base differs from fold B base
  - `alpha = 0` base-only path skips graph loading

## 2026-04-07 Phase 3 Foundation Start

### Roadmap Alignment

- Started directly from roadmap Phase 3 foundation scope:
  - domain-agnostic RL env
  - chip-specific adapter
  - simple smoke agent harness
- Did not start PPO or curriculum training yet.
- This keeps the work aligned with the roadmap's first RL deliverable slice instead of jumping ahead into optimizer work.

### New Foundation Files

- `engine/rl_env.py`
  - Gymnasium-style `reset/step/render`
  - engine owns the episode loop, adapter owns domain semantics
- `chip/rl_adapter.py`
  - chip-specific state/action/reward bridge
  - supports `analytic_hpwl`, `proxy`, `oracle`, and `proxy_with_legality_penalty`
  - random action sampling snaps to placement rows
- `chip/experiments/phase3_random_agent.py`
  - smoke harness for running a short random-policy episode and collecting a trace

### Tests

- Added:
  - `tests/engine/test_rl_env.py`
  - `tests/chip/test_rl_adapter.py`
  - `tests/chip/test_phase3_random_agent.py`
- Focused verification:
  - `pytest tests/engine/test_rl_env.py tests/chip/test_rl_adapter.py tests/chip/test_phase3_random_agent.py -q`
  - result: `10 passed`
- Wider verification:
  - `pytest tests/engine tests/chip -q`
  - result: `124 passed`

### Current Phase 3 Status

- Environment/action/reward loop now exists and is green.
- Current reward/debug path is ready for:
  - deterministic sanity baselines
  - policy-loop integration
  - later PPO training work
- PPO, curriculum scheduling, and benchmark comparison are still open.

## 2026-04-07 Phase 3 PPO Bring-Up

### Implemented

- `chip/rl_adapter.py`
  - now exposes PPO-facing observations:
    - `current_macro`
    - `current_macro_index`
    - `placed_count`
    - `graph`
    - `canvas`
    - `action_mask`
  - placement order is deterministic and fixed as `(-area, name)`
  - graph nodes now expose `placed_flag` and `current_macro_flag`
  - random action sampling respects the current action mask
- `engine/ppo_policy.py`
  - GraphSAGE-style graph trunk
  - canvas CNN trunk
  - FiLM conditioning from graph embeddings into canvas features
  - masked policy logits with `float("-inf")`
  - scalar value head
- `chip/experiments/phase3_baselines.py`
  - masked-random baseline
  - greedy one-step analytic-HPWL baseline

### Tests Added

- `tests/engine/test_ppo_policy.py`
- `tests/chip/test_phase3_baselines.py`
- expanded `tests/chip/test_rl_adapter.py`

### Verification

- focused bring-up suite:
  - `pytest tests/engine/test_rl_env.py tests/chip/test_rl_adapter.py tests/chip/test_phase3_random_agent.py tests/engine/test_ppo_policy.py tests/chip/test_phase3_baselines.py -q`
  - result: `15 passed`
- broader suite:
  - `pytest tests/engine tests/chip -q`
  - result: `129 passed`

### Current Read

- PPO-facing model inputs and masking path now exist.
- Bring-up baselines exist and can be used as pre-training checkpoints.
- The next open slice is the actual PPO rollout/update loop, not more state-space plumbing.

## 2026-04-07 Phase 3 Search + Imitation Hybrid Start

### Design Decision

- Phase 3 will not treat PPO as the only serious optimization line.
- Hybrid direction chosen:
  - search as a first-class baseline
  - imitation as a warm-start bridge
  - PPO as the RL mainline
- New design/spec files:
  - `docs/superpowers/specs/2026-04-07-phase3-search-imitation-hybrid-design.md`
  - `docs/superpowers/plans/2026-04-07-phase3-search-imitation-hybrid.md`

### Implemented In This Slice

- `chip/experiments/phase3_search.py`
  - deterministic beam-search baseline over the same masked discrete action space
  - trace-to-demonstration export helper
- `chip/imitation.py`
  - `DemonstrationStep` schema
  - JSON save/load helpers
  - demonstration-to-observation reconstruction helper
- `chip/experiments/phase3_imitation.py`
  - masked behavior-cloning cross-entropy loss
- `chip/experiments/phase3_ppo.py`
  - policy warm-start loader that tolerates a missing value head

### Key Behavioral Guarantees

- Beam search:
  - emits zero invalid actions
  - matches or beats greedy on the tiny shared benchmark
- Demonstrations:
  - round-trip through JSON without losing placement coordinates
  - can reconstruct adapter observations from stored placement snapshots
- Warm-start:
  - policy branch can be initialized without requiring a BC-trained value head

### Tests Added

- `tests/chip/test_phase3_search.py`
- `tests/chip/test_imitation.py`
- `tests/chip/test_phase3_imitation.py`
- `tests/chip/test_phase3_ppo.py`

### Verification

- targeted hybrid suite:
  - `pytest tests/chip/test_phase3_search.py tests/chip/test_imitation.py tests/chip/test_phase3_imitation.py tests/chip/test_phase3_ppo.py -q`
  - result: all green during TDD iterations
- broader suite:
  - `pytest tests/engine tests/chip -q`
  - result: `136 passed`

### Current Read

- The search/imitation/PPO interface is now a real data path, not just a design idea.
- The biggest remaining open slice is not schema or wiring anymore.
- The next valuable step is one of:
  - behavior-cloning trainer over search demonstrations
  - PPO rollout/update loop with optional BC warm-start
  - stronger search baseline such as wider beam or proxy-guided search

## 2026-04-07 Phase 3 BC + PPO Loop

### Implemented

- `chip/experiments/phase3_imitation.py`
  - behavior-cloning trainer that reuses the PPO policy architecture
  - masked supervised loss
  - tiny-dataset evaluation helper
  - BC checkpoint export that intentionally excludes `value_head.*`
- `chip/experiments/phase3_ppo.py`
  - policy initialization helper for a concrete chip problem
  - masked rollout collector using the PPO policy over the existing env/adapter
  - single-step PPO update with clipped objective
  - warm-start loader that explicitly ignores `value_head.*`

### Verified Behaviors

- tiny search demonstrations can be fit by behavior cloning with high masked accuracy
- BC checkpoints can warm-start a fresh PPO policy
- warm-started PPO still emits zero invalid actions on the tiny benchmark
- single-step PPO update returns finite `policy_loss`, `value_loss`, and entropy

### Tests

- expanded:
  - `tests/chip/test_phase3_imitation.py`
  - `tests/chip/test_phase3_ppo.py`

### Verification

- focused:
  - `pytest tests/chip/test_phase3_imitation.py tests/chip/test_phase3_ppo.py -q`
  - green during TDD iterations
- broader:
  - `pytest tests/engine tests/chip -q`
  - result: `141 passed`

### Current Read

- Phase 3 now has a minimal but real `search -> imitation -> PPO warm-start` execution path.
- The next valuable slices are no longer basic trainers or loaders.
- The highest-value next steps are:
  - multi-epoch PPO trainer over repeated rollouts
  - comparison harness: random vs greedy vs beam vs BC vs PPO-from-scratch vs PPO-warm-start
  - optional stronger search guidance such as wider beam or proxy-guided search

## 2026-04-07 Phase 3 Comparison Harness

### Implemented

- `chip/experiments/phase3_comparison.py`
  - one-call harness for:
    - masked random
    - greedy
    - beam
    - behavior cloning
    - PPO from scratch
    - PPO warm-started from BC
- `chip/experiments/phase3_ppo.py`
  - policy rollout now supports deterministic evaluation mode for stable comparisons

### Tiny-Benchmark Snapshot

- `random`
  - final HPWL `= 2.0`
- `greedy`
  - final HPWL `= 2.0`
- `beam`
  - final HPWL `= 2.0`
- `behavior_cloning`
  - final HPWL `= 2.0`
  - masked accuracy `= 1.0`
- `ppo_scratch`
  - final HPWL `= 4.0`
- `ppo_warm_start`
  - final HPWL `= 2.0`

Interpretation:

- the comparison harness now surfaces the intended bring-up story
- on the tiny benchmark, BC already recovers the search policy
- warm-start PPO reaches the beam/BC solution immediately
- scratch PPO still lags on the same setup

### Verification

- focused comparison suite:
  - `pytest tests/chip/test_phase3_comparison.py tests/chip/test_phase3_imitation.py tests/chip/test_phase3_ppo.py tests/chip/test_phase3_search.py tests/chip/test_phase3_baselines.py tests/chip/test_rl_adapter.py tests/engine/test_ppo_policy.py tests/engine/test_rl_env.py -q`
  - result: `26 passed`

### Current Read

- Phase 3 now has a direct apples-to-apples comparison surface for:
  - search
  - imitation
  - PPO from scratch
  - PPO warm-start
- The next step is no longer "can these pieces talk to each other?"
- The next step is "does warm-start remain helpful once PPO is trained for more than a single update and on larger problems?"

### Multi-Update Tiny Snapshot

- Re-ran the comparison harness with `ppo_epochs = 3` on the tiny benchmark.
- Result:
  - `beam = 2.0 HPWL`
  - `behavior_cloning = 2.0`
  - `ppo_scratch = 2.0`
  - `ppo_warm_start = 2.0`
- Interpretation:
  - after three updates, scratch PPO can catch up on the trivial benchmark
  - warm-start still shows a cleaner optimization profile:
    - scratch `value_loss ≈ 33.51`
    - warm-start `value_loss ≈ 2.30`
- This means the next useful benchmark is no longer the one-step toy.
- The next useful benchmark is a less trivial problem where sample-efficiency differences have room to show up.

### Current Stable Checkpoint

- Latest stable Phase 3 comparison checkpoint:
  - commit `183e51c`
  - message: `feat: add phase3 comparison harness`
- Last broad verification at this checkpoint:
  - `pytest tests/engine tests/chip -q`
  - result: `145 passed`

### Next Recommended Slice

- Stop using the tiny one-macro benchmark as the main decision surface.
- Move the comparison harness to a less trivial chip problem where:
  - PPO from scratch does not trivially catch up in three updates
  - BC warm-start can show a real sample-efficiency gap if it exists
- Keep the same comparison row:
  - `random`
  - `greedy`
  - `beam`
  - `behavior_cloning`
  - `ppo_scratch`
  - `ppo_warm_start`

## 2026-04-07 Phase 3 Benchmark Ladder + Sample-Efficiency Protocol

### Implemented

- `chip/experiments/phase3_benchmarks.py`
  - deterministic benchmark ladder with:
    - `tiny`
    - `small`
    - `medium`
  - each row now carries:
    - `problem`
    - `grid_shape`
    - `movable_macro_count`
    - `node_count`
    - `net_count`
- `chip/experiments/phase3_ppo.py`
  - per-update PPO history now records:
    - `final_hpwl`
    - `best_hpwl_so_far`
    - `invalid_action_count`
- `chip/experiments/phase3_comparison.py`
  - comparison harness now supports:
    - repeated `seeds`
    - per-seed rows
    - aggregate metrics
    - PPO training curves
    - `updates_to_beam_level`
    - `updates_to_bc_level`
- planning note:
  - `docs/superpowers/plans/2026-04-07-phase3-benchmark-ladder.md`

### Canonical Artifacts

- benchmark ladder summary:
  - [summary.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_benchmark_ladder/summary.json)
- full comparison reports:
  - [tiny_comparison.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_benchmark_ladder/tiny_comparison.json)
  - [small_comparison.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_benchmark_ladder/small_comparison.json)
  - [medium_comparison.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_benchmark_ladder/medium_comparison.json)

### Verification

- full suite:
  - `pytest tests/engine tests/chip -q`
  - result: `148 passed in 11.28s`

### Benchmark Ladder Results

- all three ladder benchmarks were run with:
  - `5` seeds
  - `3` PPO updates
  - the same six-way comparison row:
    - `random`
    - `greedy`
    - `beam`
    - `behavior_cloning`
    - `ppo_scratch`
    - `ppo_warm_start`

#### Tiny

- `beam = 2.0`
- `behavior_cloning = 2.0`
- `ppo_warm_start = 2.0`
- `ppo_scratch = 4.0`
- mean updates to beam level:
  - `ppo_warm_start = 1.0`
  - `ppo_scratch = 2.2`

Interpretation:

- the tiny ladder row still separates warm-start from scratch on sample efficiency
- but the task remains too easy to serve as the main experimental surface

#### Small

- `beam = 56.6`
- `behavior_cloning = 56.6`
- `ppo_warm_start = 56.6`
- `ppo_scratch = 79.92`
- mean updates to beam level:
  - `ppo_warm_start = 1.0`
  - `ppo_scratch = 4.0`

Interpretation:

- the `small` ladder row is the first non-trivial benchmark where warm-start clearly matters
- under a fixed `3`-update budget, warm-start reaches the teacher immediately
- scratch PPO does not reach the beam/BC level within the same budget

#### Medium

- `greedy = 78.5`
- `beam = 72.5`
- `behavior_cloning = 72.5`
- `ppo_warm_start = 72.5`
- `ppo_scratch = 102.6`
- mean updates to beam level:
  - `ppo_warm_start = 1.0`
  - `ppo_scratch = 4.0`

Interpretation:

- the `medium` ladder row preserves the same pattern as `small`
- beam remains a meaningfully stronger teacher than greedy
- warm-start still reaches the teacher in one update
- scratch PPO remains materially behind after three updates

### Current Read

- Phase 3 has now moved from "functional bring-up" to an experimentally meaningful ladder.
- The key evidence is not just final HPWL, but fixed-budget sample efficiency:
  - warm-start reaches beam/BC immediately on `small` and `medium`
  - scratch PPO does not catch up within `3` updates
- `value_loss` is still useful for debugging, but it is not the main success metric for this phase.
  - the robust metric is `updates_to_beam_level` under repeated seeds.
- The next valuable slice is:
  - keep the current ladder fixed
  - extend the protocol with stronger per-update curve summaries and teacher-strength sweeps
  - avoid adding new reward/proxy complexity before this comparison surface is exhausted

## 2026-04-07 Phase 3 Teacher-Strength Sweep + Closeout

### Implemented

- `chip/experiments/phase3_comparison.py`
  - comparison harness now accepts `beam_width`
- `chip/experiments/phase3_teacher_sweep.py`
  - teacher-strength sweep runner over fixed beam-width schedules
- closeout note:
  - [2026-04-07-phase3-sample-efficiency-closeout.md](C:/Users/14574/Chip/.worktrees/phase3-foundation/docs/superpowers/specs/2026-04-07-phase3-sample-efficiency-closeout.md)

### Canonical Artifacts

- teacher-strength sweep summary:
  - [summary.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_teacher_strength_sweep/summary.json)
- full sweep reports:
  - [small_teacher_sweep.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_teacher_strength_sweep/small_teacher_sweep.json)
  - [medium_teacher_sweep.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_teacher_strength_sweep/medium_teacher_sweep.json)

### Verification

- focused:
  - `pytest tests/chip/test_phase3_comparison.py tests/chip/test_phase3_teacher_sweep.py tests/chip/test_phase3_benchmarks.py tests/chip/test_phase3_ppo.py -q`
  - result: `15 passed in 11.93s`
- full:
  - `pytest tests/engine tests/chip -q`
  - result: `151 passed in 15.85s`

### Teacher-Strength Result

- `small`
  - beam width changes do not change teacher quality on the current benchmark
  - warm-start stays at teacher level
  - scratch remains behind under the same `3`-update budget
- `medium`
  - `beam_width = 1` gives `beam = 78.5`
  - `beam_width = 2` gives `beam = 72.5`
  - `beam_width = 4` matches width `2` on the current benchmark
  - warm-start matches the stronger teacher immediately
  - scratch PPO remains behind with:
    - gap to beam `= 24.1` at width `1`
    - gap to beam `= 30.1` at width `2` and `4`

### Current Read

- The sample-efficiency result now survives a teacher-strength check on the current ladder.
- On `medium`, stronger teachers widen the fixed-budget warm-start vs scratch gap rather than removing it.
- The next useful work is still experimental tightening, not new module invention:
  - plot-ready per-update artifacts
  - broader teacher sweep only where teacher quality actually changes
  - keep reward and proxy setup fixed while doing so

## 2026-04-07 Phase 3 Roadmap Closeout Slice

### Implemented

- `chip/rl_adapter.py`
  - `proxy_delta` is now a first-class reward source in `compute_reward`
  - the reward now preserves `placed_macros` when computing proxy deltas, so step-to-step deltas are real rather than silently collapsing to step `0`
- `chip/experiments/phase3_final_report.py`
  - canonical builder for a roadmap-facing Phase 3 status report
- canonical artifact:
  - [phase3_final_report.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_final_report.json)

### Tests

- added:
  - `tests/chip/test_phase3_final_report.py`
- expanded:
  - `tests/chip/test_rl_adapter.py`

### Verification

- focused:
  - `pytest tests/chip/test_rl_adapter.py tests/chip/test_phase3_final_report.py -q`
  - result: `10 passed in 1.54s`
- full:
  - `pytest tests/engine tests/chip -q`
  - result: `154 passed in 14.56s`

### Current Read

- This slice makes the current Phase 3 state easier to evaluate against the roadmap, but it does **not** fully close roadmap Phase 3.
- The canonical status in [phase3_final_report.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_final_report.json) is intentionally:
  - `roadmap_phase3_status = partial`
- Remaining roadmap gaps are now explicit rather than implicit:
  - current mainline PPO comparisons still center on the ladder/sample-efficiency surface rather than a full `RL vs SA vs analytical vs OpenROAD` table
  - routed/timing/runtime metrics are not yet unified across `3+` real chip benchmarks
  - the current production reward path is broader than pure roadmap wording because the project is using the validated production proxy/reward stack rather than claiming a stronger GNN-proxy RL result than currently exists

## 2026-04-07 Phase 3 Roadmap Eval Harness

### Implemented

- `chip/experiments/phase3_benchmarks.py`
  - `small` ladder row now exposes sample artifact paths
- `chip/experiments/phase3_roadmap_eval.py`
  - unified roadmap-facing evaluation harness over the current ladder
  - includes method rows for:
    - `random`
    - `greedy`
    - `beam`
    - `behavior_cloning`
    - `ppo_scratch`
    - `ppo_warm_start`
    - `analytical`
    - `sa`
    - `openroad_default`
- canonical artifact:
  - [phase3_roadmap_eval.json](C:/Users/14574/Chip/.worktrees/phase3-foundation/experiments/chip/phase3_roadmap_eval.json)

### Tests

- added:
  - `tests/chip/test_phase3_roadmap_eval.py`
- expanded:
  - `tests/chip/test_phase3_benchmarks.py`

### Verification

- focused:
  - `pytest tests/chip/test_phase3_roadmap_eval.py -q`
  - result: `2 passed`
- full:
  - `pytest tests/engine tests/chip -q`
  - result: `157 passed in 36.71s`

### Current Read

- The roadmap-facing eval harness now makes the missing Phase 3 pieces explicit in one table-shaped artifact instead of scattered notes.
- `analytical` and `sa` are now in the Phase 3 evaluation surface for all three current ladder benchmarks.
- `openroad_default` is present only for the sample-backed `small` benchmark and is currently marked:
  - `unsupported_missing_timing_artifacts`
- This improves roadmap alignment, but it still does not justify marking roadmap Phase 3 fully complete.
- The remaining hard gap is now very concrete:
  - extend the final table from the current `tiny/small/medium` ladder to `3+` real chip benchmarks with unified routed/timing/runtime metrics
