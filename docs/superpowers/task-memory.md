# Task Memory

## 2026-04-16 Phase 3 Warm-Start Probe Recompute

### Unit-Semantics Correction

- The earlier Ariane warm-start probe conclusion "`accepted_prefixes = 0`" was invalidated by a unit bug.
- Root cause:
  - acceptance / threshold-sweep / salvage filtering treated `hpwl_threshold = 2.85` as raw HPWL
  - actual rollout HPWL is stored in raw units around `2_800_000+`
- Fix is now in:
  - [phase3_macroplacement_ppo.py](C:/Users/14574/Chip/chip/experiments/phase3_macroplacement_ppo.py)
- New invariant:
  - small thresholds like `2.85` are interpreted as `2.85M`
  - raw thresholds like `2_850_000` remain valid
  - artifacts now also record `hpwl_threshold_raw` / `hpwl_threshold_sweep_raw`
- Fresh verification:
  - `python -m pytest tests/engine tests/chip -q`
  - `222 passed in 26.79s`

### Corrected Warm-Start Probe Artifacts

- original probe:
  - [phase3_ariane_warm_start_probe.json](C:/Users/14574/Chip/experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe.json)
- corrected acceptance-only recompute:
  - [phase3_ariane_warm_start_probe_recomputed_acceptance.json](C:/Users/14574/Chip/experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe_recomputed_acceptance.json)
- corrected full recompute with salvage:
  - [phase3_ariane_warm_start_probe_recomputed_full.json](C:/Users/14574/Chip/experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe_recomputed_full.json)

### Corrected Headline Numbers

- `update0` warm-start cohort:
  - `completion_rate = 0.36`
  - `median final_hpwl = 2.9494M`
  - `mean final_hpwl = 2.9350M`
  - `hpwl_eligible @ 2.85M = 5 / 25`
  - `accepted_prefixes @ 2.85M + remaining [8,13] = 1`
  - threshold sweep:
    - `2.88M + [8,13] -> 2`
    - `2.90M + [8,13] -> 2`
    - `2.92M + [8,13] -> 2`

- `update1` cohort after a single PPO update:
  - `completion_rate = 0.08`
  - `median final_hpwl = 3.0401M`
  - `mean final_hpwl = 3.0159M`
  - `hpwl_eligible @ 2.85M = 2 / 25`
  - `accepted_prefixes @ 2.85M + remaining [8,13] = 0`
  - threshold sweep:
    - `2.88M + [8,13] -> 1`
    - `2.90M + [8,13] -> 1`
    - `2.92M + [8,13] -> 1`

- corrected salvage calibration over all non-completed rows with `hpwl <= 2.85M`:
  - `total_candidates = 5`
  - `greedy_completable = 1`
  - `random_completable = 0`
  - `recommended_salvage_mode = greedy_or_random`

### Current Interpretation

- Replay already confirmed the `2.8127M` warm-start rollout is real, not a logging artifact.
- The corrected warm-start probe changes the framing:
  - the main problem is no longer "can we generate any good prefixes?"
  - the main problem is "PPO updates are actively degrading an already capable warm-start policy"
- The strongest current evidence is the completion-rate collapse:
  - `update0 = 0.36`
  - `update1 = 0.08`
  - later canonical PPO had already dropped to `~0.02`
- This now supports a methodology pivot:
  - reverse curriculum is no longer the only or primary next hypothesis
  - KL-anchored / behavior-preserving PPO should be considered first-class next work

### Process Invariants Added

- Any experiment that reports HPWL thresholds must preserve both:
  - user-facing shorthand (e.g. `2.85`)
  - normalized raw threshold (`2_850_000`)
- Any long-running experiment should preserve:
  - per-update rollout trace
  - per-rollout replayable `action_indices`
- This was reinforced by two concrete failures:
  - earlier missing per-rollout canonical data blocked reuse
  - threshold-unit mismatch silently invalidated acceptance counts

## 2026-04-08 Phase 3 Ariane Bring-Up

### Milestone A Status

- `Milestone A` is now `pass`.
- Root cause of the earlier `A3` failure was identified via control-variable probing:
  - the sequential Ariane env was treating **future movable macros at their seed locations as active obstacles**
  - this made `build_action_mask()` far too restrictive and caused artificial dead-ends
- Evidence before the fix:
  - current semantics:
    - `valid_actions_step0 = 160`
    - greedy dead-ended after `50` steps
    - greedy improvement over masked-random mean HPWL was only `~10.97%`
- Evidence from the probe after switching to `fixed + placed + current` semantics:
  - step `0` valid actions jumped from `160` to `928`
  - active-only greedy reached `105` steps with `final_hpwl ≈ 1.6826M`
- Implementation change:
  - `chip/rl_adapter.py`
    - legality / action-mask / canvas occupancy now treat only:
      - fixed macros
      - already placed macros
      - the current macro candidate
    - future movable macros no longer reserve space before they are placed
- Supporting tests were added/updated in:
  - [test_rl_adapter.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tests/chip/test_rl_adapter.py)
  - [test_phase3_baselines.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tests/chip/test_phase3_baselines.py)

### Updated Ariane Milestone A Artifacts

- masked-random artifact:
  - [phase3_masked_random_ariane.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_masked_random_ariane.json)
- greedy artifact:
  - [phase3_greedy_ariane.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_greedy_ariane.json)

Current canonical numbers:

- reward pipeline:
  - `single_step_reward_seconds_mean ≈ 0.00353s`
  - still comfortably below the `50ms` gate
- masked random:
  - `100` episodes
  - `mean final_hpwl ≈ 3.2218M`
  - `mean runtime ≈ 38.17s`
  - `completed_episode_rate = 0.0`
  - typical dead-end depth moved from the earlier `~50` range to roughly `80-90` steps
- greedy one-step analytic HPWL:
  - `final_hpwl ≈ 1.6826M`
  - `steps = 105`
  - `termination_reason = no_valid_actions`
  - improvement over masked-random mean HPWL:
    - `~47.77%`
  - `passes_20_percent_gate = true`

Interpretation:

- `Milestone A1`: pass
- `Milestone A2`: pass
- `Milestone A3`: pass
- Ariane `32x32` still does not fully complete the `133`-macro sequence under random or greedy, but the user-defined gate was the `>=20%` greedy improvement check, and that gate is now satisfied.
- Dead-end handling remains a known property of the current formulation and must be treated as a normal termination mode during PPO bring-up.

### Milestone B1 Status

- New Ariane PPO bring-up entry point:
  - [phase3_macroplacement_ppo.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/chip/experiments/phase3_macroplacement_ppo.py)
- Supporting generic PPO helpers were added in:
  - [phase3_ppo.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/chip/experiments/phase3_ppo.py)
- New capabilities:
  - `uniform_legal_policy` rollout mode for plumbing checks
  - fixed-rollout replay for single-episode overfit checks
  - rollout metrics now include:
    - `episode_return`
    - `termination_reason`
    - `completed_episode`
    - `is_legal`
    - `approx_kl`
- Fresh verification after these additions:
  - `pytest tests/engine tests/chip -q`
  - `184 passed in 18.19s`

### Milestone B1 Ariane Smoke

- uniform-policy plumbing smoke artifact:
  - [phase3_uniform_policy_plumbing_ariane.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_bringup/phase3_uniform_policy_plumbing_ariane.json)
- fixed-episode overfit smoke artifact:
  - [phase3_overfit_single_episode_ariane.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_bringup/phase3_overfit_single_episode_ariane.json)

Smoke results:

- uniform-policy plumbing (`3` updates):
  - did not crash
  - all policy/value losses finite
  - invalid action count remained `0`
- fixed-episode overfit (`10` updates):
  - `final_return_improved = true`
  - overfit sanity therefore points in the correct direction

Interpretation:

- `Milestone B1` smoke is positive.
- Two formal `100`-update runs were launched after the smoke:
  - `plumbing`
  - `overfit`
- At the time of this memory update they were still running in the background, so `Milestone B1` formal status is:
  - `in_progress`

### Current Next Entry Point

- Do **not** start Milestone C or expand to `BlackParrot / MemPool` yet.
- Wait for the formal `100`-update `Milestone B1` Ariane runs to finish.
- If they pass:
  - proceed to `Milestone B2` with the Ariane `2000`-update PPO checkpoint
- If they fail:
  - stop and debug under the user’s milestone discipline instead of pushing forward blindly

## 2026-04-08 Phase 3 MacroPlacement Switch

### Current Branch State

- Active worktree: [phase3-macroplacement](C:/Users/14574/Chip/.worktrees/phase3-macroplacement)
- Active branch: `codex/phase3-macroplacement`
- MacroPlacement benchmark repo cloned at:
  - [tmp/MacroPlacement](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement)
- Fresh verification after the Ariane-first benchmark switch scaffolding:
  - `pytest tests/engine tests/chip -q`
  - `163 passed in 15.85s`

### Problem-Formulation Decision

- The `mgc_*` benchmarks remain valid for:
  - `Phase 1` acceptance
  - `Phase 2` proxy validation
- They are not appropriate for `Phase 3` macro-placement RL.
  - The earlier `mgc_*` probe established that these are standard-cell-dominant designs with `30k+ / 30k+ / 112k+` movable standard cells.
  - This is a problem-formulation mismatch, not a data-quality issue.
- `Phase 3` therefore switches to true macro-placement benchmarks from TILOS MacroPlacement, starting with `Ariane` only.

### Ariane Asset Discovery

- Ariane-first benchmark spec is implemented in:
  - [phase3_macroplacement_benchmarks.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/chip/experiments/phase3_macroplacement_benchmarks.py)
- Current first benchmark:
  - `variant = ariane133`
  - `platform = NanGate45`
  - benchmark name: `ariane133_ng45`
- Key discovered assets:
  - reference macro-placement DEF:
    - [ariane133_fp_placed_macros.def](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Flows/NanGate45/ariane133/def/ariane133_fp_placed_macros.def)
  - floorplan DEF:
    - [ariane133_fp.def](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Flows/NanGate45/ariane133/def/ariane133_fp.def)
  - manual human floorplan DEF:
    - [manual_floorplan.def](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Flows/NanGate45/ariane133/def/manual_floorplan.def)
  - constraints:
    - [ariane.sdc](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Flows/NanGate45/ariane133/constraints/ariane.sdc)
  - enablement LEFs / liberty:
    - [Enablements/NanGate45/lef](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Enablements/NanGate45/lef)
    - [Enablements/NanGate45/lib](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Enablements/NanGate45/lib)

### Ariane Feasibility Probe

- Probe implementation is in:
  - [phase3_macroplacement_probe.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/chip/experiments/phase3_macroplacement_probe.py)
- ORFS completion wrapper is in:
  - [phase3_macroplacement_orfs.py](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/chip/experiments/phase3_macroplacement_orfs.py)
- Canonical Ariane status artifact:
  - [ariane133_ng45_status.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_probe/ariane133_ng45_status.json)
- Canonical Ariane ORFS completion artifact:
  - [ariane133_ng45_reference_orfs_completion.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_probe/ariane133_ng45_reference_orfs_completion.json)
- Key probe numbers on `ariane133_ng45` with `32x32` grid:
  - `movable_macro_count = 133`
  - `single_legality_check_seconds ≈ 0.000865`
  - `build_action_mask_seconds ≈ 0.2665`
  - `valid_action_fraction = 0.15625`
- Interpretation:
  - The legality-performance gate is healthy on Ariane.
  - Unlike the earlier `mgc_*` full-design probe, Ariane does **not** immediately fail on action-mask cost.
  - Under the current implementation, Ariane no longer appears blocked by legality latency.

### Ariane Gate Status

- The current structured gate state in [ariane133_ng45_status.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_probe/ariane133_ng45_status.json) is:
  - `legality_perf = pass`
    - measured `0.000865s`
    - explicit gate `<= 0.01s`
    - about `11.6x` headroom
  - `action_mask_perf = pass`
    - measured `0.2665s` on `32x32`
  - `reference_def_routability = pass`
    - via `ORFS completion`
    - with `FLOORPLAN_DEF`
    - and `SKIP_CTS_REPAIR_TIMING=1`
    - `runtime_minutes ≈ 70.77`
    - `routing_overhead_ratio ≈ 1.088`
  - `rl_bring_up = not_started`

### Pure Eval Status

- A generic `read_def + injected placement + detailed_placement + global_route` eval was attempted on:
  - [ariane133_fp_placed_macros.def](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/tmp/MacroPlacement/Flows/NanGate45/ariane133/def/ariane133_fp_placed_macros.def)
- Current result:
  - `status = blocked_generic_eval`
- Evidence in the status artifact shows:
  - OpenROAD successfully reads the LEFs and the macro DEF.
  - The DEF contains `133` components and `495` pins, i.e. the macro floorplan itself.
  - `detailed_placement` reports `0` movable cells and `0.0u` HPWL.
  - The run then fails with:
    - `EST-0005 Run global_route before estimating parasitics for global routing`
- Interpretation:
  - `ariane133_fp_placed_macros.def` is a macro-floorplan initialization artifact, not a complete routed-flow input for the current generic eval harness.
  - This generic path remains blocked and is now explicitly distinct from the ORFS completion path below.

### ORFS Completion Status

- Ariane reference floorplan is now proven routable through the narrow ORFS completion path.
- The working ORFS path uses:
  - built-in Ariane config from ORFS
  - `FLOORPLAN_DEF=<ariane133_fp_placed_macros.def>`
  - `CORE_UTILIZATION=` blanked to avoid conflicting floorplan generation
  - `SKIP_CTS_REPAIR_TIMING=1` as a narrow CTS workaround
- The canonical completion artifact is:
  - [ariane133_ng45_reference_orfs_completion.json](C:/Users/14574/Chip/.worktrees/phase3-macroplacement/experiments/chip/phase3_macroplacement_probe/ariane133_ng45_reference_orfs_completion.json)
- Key completion metrics:
  - `placement_source = ariane_reference_floorplan_orfs_completion`
  - `final_hpwl = 7126309.775001906`
  - `total_routed_wirelength = 7751289`
  - `routing_overhead_ratio ≈ 1.088`
  - `drc_violations = 0`
  - `wns = -0.676384`
  - `tns = -1473.26`
  - `runtime_seconds = 4246.43` (`file_timestamps`)
- ORFS completion mode is explicitly:
  - `fast`
  - because `SKIP_CTS_REPAIR_TIMING=1` is enabled
- Timing interpretation:
  - `WNS/TNS` are reported but **not gated**
  - this is intentional for fast, repeatable macro-placement completion
  - signoff-quality timing comparison is deferred until a later final-eval slice
- Reward interpretation:
  - Ariane ORFS completion is **offline eval only**
  - `~70.8` minutes is far too slow for PPO terminal reward during bring-up
  - PPO bring-up should continue using:
    - `seed-completed proxy delta`
    - `analytic HPWL delta`
    - `legality penalty`
- Interpretation:
  - Ariane now passes both the legality-performance gate and the reference-floorplan routability gate.
  - The reference floorplan now acts as a credible ceiling baseline:
    - clean routing (`DRC = 0`)
    - low routing overhead (`~8.8%`)
  - The remaining work is benchmark harness integration and later RL bring-up, not more ORFS root-cause hunting.

### Next Entry Point

- Keep `Ariane133-NanGate45` as the first macro-placement benchmark.
- Next action is **not** more PPO work yet.
- The next narrow step is:
  - package the proven Ariane ORFS completion path into the macro-placement harness so the benchmark can emit `HPWL / routed WL / WNS / TNS / runtime` under the canonical Phase 3 protocol
- Only after that should the benchmark ladder expand to:
  - `BlackParrot`
  - `MemPoolGroup`

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

## 2026-04-08 Ariane Reward-Hacking Debug

### Investigation

- Reproduced the bad `B1` behavior on Ariane:
  - `phase3_overfit_single_episode_ariane.json` showed entropy collapsing to `0`
  - final HPWL worsened while episode return improved
  - termination stayed `no_valid_actions`
- Control-variable probes isolated three distinct causes:
  - `Phase 2` frozen linear proxy is badly OOD on macro-placement Ariane
    - training target range: about `0.39M .. 2.53M`
    - Ariane seed prediction: `317.8M`
    - Ariane first-step prediction: `-786.8M`
    - first-step proxy delta: `-1.10B`
    - Ariane first-step `displacement_max_sites`: `3400.6` vs Phase 2 training max about `361`
  - per-episode reward normalization destroyed reward semantics for Ariane bring-up
    - with `w_proxy=0` but normalization still on, masked-random `episode_return` vs final HPWL correlation stayed near `0.05`
    - with `w_proxy=0` and normalization off, the same correlation became effectively `-1.0`
  - `collect_rollout_for_action_trace()` incorrectly marked exhausted partial traces as `completed`
    - this made the fixed-rollout teacher artifact look cleaner than it really was

### Fixes

- `chip/experiments/phase3_macroplacement_bringup.py`
  - Ariane PPO bring-up reward now uses:
    - `w_proxy = 0.0`
    - `w_hpwl = 1.0`
    - `normalize_rewards = False`
  - rationale:
    - Phase 2 proxy remains valid for Phase 2
    - it is not valid as an online reward term on macro-placement Ariane
- `chip/experiments/phase3_ppo.py`
  - `collect_rollout_for_action_trace()` now distinguishes:
    - `completed`
    - `no_valid_actions`
    - `truncated`
    - `trace_exhausted`
  - it no longer silently marks partial traces as completed

### Regression Tests

- added:
  - `tests/chip/test_phase3_macroplacement_bringup.py`
  - `tests/chip/test_phase3_ppo_regressions.py`
- new guards:
  - Ariane reward builder keeps `w_proxy = 0.0`, `w_hpwl = 1.0`, `normalize_rewards = False`
  - partial action traces are not reported as completed

### Post-Fix Read

- reward-hacking is fixed in the important sense:
  - return no longer explodes to `1e9 .. 1e10`
  - final HPWL no longer gets worse as return improves
- updated Ariane overfit artifact:
  - initial eval HPWL: about `3.1705M`
  - final eval HPWL after `40` updates: about `3.1410M`
  - entropy still drops in fixed-rollout overfit, but this now behaves like expected deterministic fitting rather than reward exploitation
- updated uniform-policy plumbing artifact:
  - losses remain finite
  - entropy stays high (`~5.1`)
  - no invalid actions

### Current Judgment

- `reward-hacking` was caused by an interaction between:
  - OOD proxy magnitudes
  - episode normalization
  - dead-end-prone action space
- after the fix, the remaining entropy drop in the fixed-rollout overfit check is not the same bug:
  - it reflects deterministic memorization pressure in the overfit sanity test
  - not a reward exploit
- next Phase 3 milestone should continue from the repaired Ariane reward setup, not the old proxy-driven one

## 2026-04-09 Ariane BC / Warm-Start Milestone B Update

### Root Cause

- Ariane behavior cloning was not numerically unstable on its own; the real bug was trace semantics.
- `run_greedy_episode()` and `run_masked_random_episode()` were recording `step_index` after `env.step()`, i.e. post-action.
- `build_demonstrations_from_trace()` interpreted `step_index` as pre-action and rebuilt the observation at the wrong decision point.
- This made teacher actions look illegal inside their reconstructed observations:
  - `demo_count = 128`
  - `invalid_target_count = 52`
  - `all_invalid_row_count = 51`
- Because masked cross-entropy was being asked to learn actions that were already masked out, Ariane BC produced:
  - `masked_accuracy ~= 0.0857`
  - `final_loss = NaN`

### Fixes

- `chip/experiments/phase3_baselines.py`
  - episode traces now record pre-action `step_index`
- `chip/experiments/phase3_search.py`
  - `build_demonstrations_from_trace()` now derives `step_index` from replay state (`len(state.placed_macros)`) instead of trusting external trace metadata
- added regression coverage:
  - `tests/chip/test_phase3_imitation.py`
    - greedy demonstrations preserve valid teacher actions
  - `tests/chip/test_phase3_baselines.py`
    - greedy trace uses pre-action step indices

### Verification

- focused:
  - `python -m pytest tests/chip/test_phase3_baselines.py tests/chip/test_phase3_imitation.py -q`
  - result: `11 passed in 2.49s`
- full:
  - `python -m pytest tests/engine tests/chip -q`
  - result: `190 passed in 18.73s`

### Post-Fix Read

- the teacher/demo pipeline is now healthy:
  - `invalid_target_count = 0`
  - `all_invalid_row_count = 0`
  - `masked_accuracy = 1.0`
  - `final_loss = 0.0`
- deterministic BC rollout on Ariane is materially better than masked random but still above greedy:
  - `bc_rollout_hpwl ~= 3.0026M`
- quick `5`-update PPO comparison on the repaired setup:
  - `scratch_5_final_hpwl ~= 3.0853M`
  - `warm_5_final_hpwl ~= 3.1360M`
  - `warm-start` did not immediately improve on scratch under the current PPO update regime
- interpretation:
  - Milestone B is no longer blocked by bad demonstrations
  - the current blocker has shifted to PPO update dynamics on Ariane, not BC data quality

### Current Status

- a longer Ariane comparison run is in flight:
  - artifact target:
    - `experiments/chip/phase3_macroplacement_compare_100/phase3_ariane_compare_100.json`
  - compares:
    - deterministic BC rollout
    - `ppo_scratch_100`
    - `ppo_warm_start_100`
- this is the current Milestone B gate readout path; do not restart from the old `phase3_macroplacement_warmstart_pilot.json`

## 2026-04-09 Ariane Milestone B2 Training-Dynamics Diagnosis

### First checkpoint readout

- `phase3_macroplacement_compare_100/phase3_ariane_compare_100.json` is now complete.
- Current same-protocol Ariane comparison:
  - `bc_rollout_hpwl ~= 3.0026M`
  - `ppo_scratch_100 final_hpwl ~= 3.1507M`
  - `ppo_warm_start_100 final_hpwl ~= 3.3519M`
- Both PPO variants fail the current B2 gate.
- Warm-start is worse than scratch at `100` updates under the current training dynamics.

### What the curves say

- `scratch`
  - starts high-entropy (`~5.17`)
  - improves gradually and reaches best rollout around update `81`
    - `best_hpwl ~= 3.0346M`
  - then degrades again as entropy collapses toward `0`
- `warm-start`
  - starts from a much better first rollout
    - update `1 hpwl ~= 2.8579M`
  - but this is also its best rollout
  - degrades almost immediately across the next few updates
  - entropy starts low (`~0.65`) and collapses to near-zero by about update `10`

### Control-variable probes

- Probe 1: remove value loss (`value_coef = 0.0`) for `20` updates
  - `scratch`
    - best rollout improves to about `3.0147M`
    - final entropy stays much higher (`~4.17`)
  - `warm-start`
    - still degrades, but entropy collapse is less severe
    - final entropy stays around `0.12` instead of numerical zero
- Interpretation:
  - giant value loss on raw returns is a real negative contributor
  - shared-trunk PPO is being pulled around by value regression at Ariane reward scale

- Probe 2: keep `value_coef = 0.0`, raise entropy bonus for warm-start
  - `entropy_coef = 0.1`
    - final hpwl after `10` updates: about `3.0637M`
    - final entropy: about `0.92`
  - `entropy_coef = 1.0`
    - final hpwl after `10` updates: about `2.9341M`
    - final entropy: about `0.72`
- Interpretation:
  - warm-start degradation is not just "PPO cannot learn"
  - the current training dynamics are over-exploiting a low-entropy BC initialization
  - stronger entropy regularization materially preserves the good warm-start region

### Current read

- B2 is now narrowed to a training-scale problem:
  - `value loss magnitude` is too large for the current shared trunk
  - `entropy regularization` is too weak for BC-warm-started policies
- This is not a reward bug and not a demonstration bug anymore.
- The next minimal fix should change PPO update dynamics, not benchmark/data plumbing.

## 2026-04-09 Ariane B2 Tuned PPO Config

### Decision

- Ariane checkpoint training now has an explicit tuned config:
  - `init_seed = 0`
  - `learning_rate = 1e-3`
  - `value_coef = 0.0`
  - `entropy_coef = 1.0`
- rationale:
  - `value_coef = 0.0` removes the large shared-trunk value regression term that was destabilizing both scratch and warm-start
  - `entropy_coef = 1.0` preserves exploration and stops BC warm-start from collapsing immediately into a bad low-entropy region
  - `init_seed = 0` makes Ariane checkpoint artifacts reproducible instead of depending on policy initialization noise

### Code / Test Status

- `chip/experiments/phase3_macroplacement_ppo.py`
  - now exposes `build_ariane_checkpoint_training_config()`
  - `run_ariane_ppo_checkpoint()` seeds torch and records `training_config` in artifact metadata
- regression:
  - `tests/chip/test_phase3_macroplacement_ppo.py`
  - verifies:
    - `value_coef == 0.0`
    - `entropy_coef == 1.0`
    - `init_seed == 0`
    - config is forwarded into `train_ppo_policy()`

### Verification

- focused:
  - `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py tests/engine tests/chip -q`
  - result: `23 passed`

### Tuned 10-update probe

- artifact:
  - `experiments/chip/phase3_b2_tuned_compare_10/phase3_b2_tuned_compare_10.json`
- results:
  - `bc_rollout_hpwl ~= 3.0026M`
  - `ppo_scratch_10 final ~= 3.1779M`, `best ~= 3.0832M`
  - `ppo_warm_start_10 final ~= 2.9341M`, `best ~= 2.8579M`
- interpretation:
  - under the tuned config, warm-start is again better than scratch at equal budget
  - the warm-start policy still drifts away from its initial best rollout, but it no longer catastrophically degrades the way it did under the old config

### Next action

- use the tuned config for the next `compare_100` Ariane run
- do not interpret old `phase3_macroplacement_compare_100/phase3_ariane_compare_100.json` as the current B2 status once the tuned artifact exists
- 2026-04-09 B2 dynamics diagnosis update:
  - Added `advantage normalization` in `chip/experiments/phase3_ppo.py`.
  - Regression tests:
    - `tests/chip/test_phase3_ppo.py`
    - `python -m pytest tests/chip/test_phase3_ppo.py tests/chip/test_phase3_macroplacement_ppo.py -q`
    - `13 passed`
  - Full suite after this change:
    - `python -m pytest tests/engine tests/chip -q`
    - `193 passed in 22.84s`
  - Ariane short-run artifact after advantage normalization:
    - `experiments/chip/phase3_b2_advnorm_compare_10/phase3_b2_advnorm_compare_10.json`
    - `scratch final_hpwl ~= 3.1078M`
    - `warm-start final_hpwl ~= 3.1464M`
    - Interpretation: normalization helps scratch but does not by itself preserve warm-start quality.
  - Warm-start learning-rate probe:
    - `experiments/chip/phase3_b2_warm_lr1e4_10/phase3_b2_warm_lr1e4_10.json`
    - Compared against `phase3_b2_advnorm_compare_10.json`:
      - `BC rollout ~= 3.0026M`
      - `scratch @ 1e-3 ~= 3.1078M`
      - `warm-start @ 1e-3 ~= 3.1464M`
      - `warm-start @ 1e-4 ~= 2.9768M`
      - `warm-start @ 1e-4 best-history hpwl ~= 2.8311M`
    - Current diagnosis:
      - warm-start drift is now best explained by update size mismatch rather than reward or demonstration quality
      - scratch and warm-start should not share the same learning-rate default in Ariane B2 bring-up

## 2026-04-10 Ariane B2 Termination Diagnosis

### Current checkpoint position

- canonical Ariane PPO checkpoint artifact:
  - `experiments/chip/phase3_macroplacement_checkpoint_2000/phase3_ppo_ariane_checkpoint.json`
- key numbers:
  - `final_hpwl ~= 2.8402M`
  - `legality_rate = 1.0`
  - `completed_episode_rate = 0.0`
  - `termination_reason = no_valid_actions`
  - `steps = 85`

### B2 gate interpretation

- with canonical baselines:
  - masked random mean HPWL: `3.2218M`
  - BC rollout HPWL: `3.0026M`
  - greedy HPWL: `1.6826M`
- `scratch @ 1e-3, 100 updates` and the later `warm-start` checkpoint both remain below the B2 quality gate:
  - not `>= 30%` better than masked random
  - not close to `80%` of the greedy gap

### Separated-config result

- artifact:
  - `experiments/chip/phase3_b2_separated_compare_200/phase3_b2_separated_compare_200.json`
- final results:
  - `BC rollout ~= 3.0026M`
  - `scratch @ 1e-3 ~= 3.4000M`
  - `warm + entropy x5 @ 1e-4 ~= 2.9677M`
  - `warm + value-prefit @ 1e-4 ~= 3.0888M`
- best-history results:
  - `scratch best ~= 3.0222M`
  - `warm + entropy x5 best ~= 2.7183M`
  - `warm + value-prefit best ~= 2.8311M`
- interpretation:
  - warm-start is still useful on Ariane
  - the best current PPO recipe is:
    - warm-start from BC
    - `learning_rate = 1e-4`
    - `entropy_coef = 5.0`
  - value-prefit helps somewhat but is not the strongest fix

### Failure-mode diagnosis

- a full retraining-based termination audit was started, but it was too expensive to keep on the critical path
- instead, cheaper probes were used to separate policy failure from formulation limits

- probe 1: map PPO failure step onto greedy trace
  - PPO fails at step `85`
  - failing macro:
    - `i_cache_subsystem/i_nbdcache/sram_block[3].tag_sram/macro_mem[0].i_ram`
  - remaining macros at PPO failure: `48`
  - on the greedy trace at the same step, that same macro still has `104` legal actions
  - interpretation:
    - PPO is not hitting a hard "physically no legal slots exist anywhere" wall at step 85
    - PPO is reaching a policy-induced dead-end earlier than greedy

- probe 2: greedy tail audit
  - greedy itself does **not** complete under the current formulation
  - greedy stops at step `105`
  - termination reason: `no_valid_actions`
  - remaining macros: `28`
  - failing macro:
    - `i_cache_subsystem/i_nbdcache/sram_block[5].data_sram/macro_mem[6].i_ram`
  - legal-action counts in the last 25 greedy steps shrink from `135 -> 6`
  - interpretation:
    - the current `32x32 + fixed descending-area order` formulation has a real late-stage packing cliff
    - greedy is a useful heuristic ceiling, but it is **not** a completion ceiling under the current action formulation

### Current diagnosis summary

- there are two separate blockers now:
  1. **policy divergence blocker**
     - PPO dies much earlier than greedy
     - this is a training-dynamics / exploration-quality issue
  2. **formulation ceiling**
     - even greedy eventually dies with `no_valid_actions`
     - this points to action-space / ordering / late-stage packing limits

- therefore:
  - Ariane B2 is **not** a reward bug anymore
  - it is **not** a legality bug anymore
  - it is now a combination of:
    - PPO policy quality not preserving enough optionality
    - current `32x32` fixed-order formulation not being strong enough to finish all 133 macros

### Implication for next work

- do not assume "pass B2 by tuning PPO harder" is sufficient
- the next meaningful decision point is whether to:
  - improve policy against the current formulation, or
  - explicitly treat the formulation ceiling as the next object of work

## 2026-04-10 Ariane formulation diagnosis and promotion

### Greedy formulation sweep

- artifact:
  - `experiments/chip/phase3_macroplacement_formulation_sweep/phase3_ariane_formulation_sweep.json`
- sweep axes:
  - grid: `32x32`, `64x64`
  - macro order: `descending_area`, `seed_yx`
- results:
  - `g32x32_descending_area`
    - `steps = 105`
    - `termination = no_valid_actions`
    - `final_hpwl ~= 1.6826M`
  - `g32x32_seed_yx`
    - `steps = 105`
    - `termination = no_valid_actions`
    - `final_hpwl ~= 1.8306M`
  - `g64x64_descending_area`
    - `steps = 133`
    - `termination = completed`
    - `final_hpwl ~= 1.0913M`
  - `g64x64_seed_yx`
    - `steps = 133`
    - `termination = completed`
    - `final_hpwl ~= 1.2678M`

### Interpretation

- the dominant formulation lever is **grid resolution**, not macro order
- `seed_yx` is consistently worse than `descending_area`
- `64x64 + descending_area` is the best current Ariane formulation:
  - completion restored under greedy
  - HPWL improves by about `0.591M` relative to the old canonical `32x32 + descending_area`
  - this is the first formulation result that turns the late-stage packing cliff from "ceiling" into "avoidable under the same heuristic"

### Performance gate for 64x64

- quick mask probe:
  - `32x32 build_action_mask ~= 0.144s`
  - `64x64 build_action_mask ~= 0.585s`
- interpretation:
  - `64x64` is about `4x` slower per mask, as expected
  - but it is still sub-second and remains practical for Ariane bring-up

### PPO apples-to-apples short checkpoint

- artifacts:
  - `experiments/chip/phase3_macroplacement_checkpoint_32x32_20/phase3_ppo_ariane_checkpoint.json`
  - `experiments/chip/phase3_macroplacement_checkpoint_64x64_20/phase3_ppo_ariane_checkpoint.json`
- same PPO recipe, `20` updates, warm-start canonical config:
  - `32x32`
    - `final_hpwl ~= 3.0277M`
    - `steps = 85`
    - `completed_episode_rate = 0.0`
  - `64x64`
    - `final_hpwl ~= 2.9962M`
    - `steps = 118`
    - `completed_episode_rate = 0.1`
- interpretation:
  - formulation promotion helps PPO too, not just greedy
  - the gain is modest in HPWL at `20` updates, but substantial in survivability:
    - `+33` extra placed macros
    - first nonzero completion rate under PPO

### Canonical Ariane formulation

- Ariane PPO bring-up now defaults to:
  - `grid_shape = (64, 64)`
  - `macro_order_strategy = descending_area`
- code:
  - `chip/experiments/phase3_macroplacement_bringup.py`
    - `build_ariane_formulation_config()`
  - `chip/experiments/phase3_macroplacement_ppo.py`
    - Ariane runners resolve `grid_shape` from the canonical formulation when not explicitly overridden

## 2026-04-10 / 2026-04-11 Canonical64 B2 checkpoint

### 100-update checkpoint result

- artifact:
  - `experiments/chip/phase3_macroplacement_checkpoint_64x64_100/phase3_ppo_ariane_checkpoint.json`
- run status:
  - completed successfully
  - no lingering training process remains for this checkpoint
- key numbers:
  - `final_hpwl ~= 2.9421M`
  - `best_hpwl ~= 2.8127M`
  - `steps = 129`
  - `termination_reason = no_valid_actions`
  - `completed_episode_rate = 0.02`
  - `legality_rate = 1.0`
  - `entropy_final ~= 6.01`
  - `entropy_min ~= 1.41`

### Interpretation

- canonical `64x64` shows a **real upward shift** relative to the old formulation:
  - compared with `64x64 @ 20 updates`, HPWL improves from about `2.9962M -> 2.9421M`
  - survivability also improves from `118 -> 129` steps
- however, `Milestone B2` is still **not passed**
  - completion remains rare (`2%`)
  - rollout still dies via `no_valid_actions`
  - PPO quality is improved, but still far from the greedy ceiling

### Important bookkeeping note

- the checkpoint artifact's `reference_levels` fields were serialized as `0`
- do **not** trust the auto-computed gate booleans in this artifact as the final B2 verdict
- compare against the separately established Ariane baselines instead:
  - masked random:
    - `experiments/chip/phase3_masked_random_ariane.json`
  - greedy:
    - `experiments/chip/phase3_greedy_ariane.json`
  - formulation sweep:
    - `experiments/chip/phase3_macroplacement_formulation_sweep/phase3_ariane_formulation_sweep.json`

### Current B2 status

- status: `in_progress / not passed`
- current best formulation-backed PPO evidence:
  - canonical `64x64 + descending_area` is better than `32x32`
  - but PPO still primarily fails by late-stage `no_valid_actions`
- next likely work:
  - fix `reference_levels` serialization in checkpoint artifacts
  - continue B2 diagnosis from the new canonical64 baseline rather than reopening old 32x32 runs

## 2026-04-11 / 2026-04-12 completion-bonus counterfactual

### Why this probe was run

- canonical `64x64 @ 100 updates` showed:
  - `max steps = 133`
  - `completed_episode_rate = 0.02`
- this means PPO already occasionally reaches full completion on Ariane, but too rarely for that signal to dominate training
- hypothesis:
  - a terminal completion bonus might improve credit assignment and increase completion frequency

### Probe setup

- artifact:
  - `experiments/chip/phase3_macroplacement_checkpoint_64x64_20_completion_bonus/phase3_ppo_ariane_checkpoint.json`
- config:
  - canonical Ariane formulation: `64x64 + descending_area`
  - `20` updates
  - reward override:
    - `w_proxy = 0.0`
    - `w_hpwl = 1.0`
    - `legality_penalty = 10.0`
    - `completion_bonus = 250000.0`

### Result

- final checkpoint:
  - `final_hpwl ~= 2.9245M`
  - `completed_episode_rate = 0.05`
  - `remaining_macro_count = 13`
- comparison vs baseline `64x64 @ 20 updates`:
  - baseline:
    - `final_hpwl ~= 2.9962M`
    - `completed_episode_rate = 0.10`
  - completion bonus:
    - slightly better aggregate HPWL
    - **worse completion rate**
- completed-only subset:
  - exactly `1` completed episode in the 20-update history
  - completed HPWL `~= 3.2047M`
  - this is materially worse than the current best non-completed rollouts
- non-completed subset:
  - `19` episodes
  - best HPWL `~= 2.8127M`
  - median HPWL `~= 3.1538M`

### Interpretation

- `completion_bonus = 250k` does **not** fix the B2 bottleneck
- it produces the bad failure mode we explicitly wanted to check for:
  - completion happens slightly/rarely
  - but completed rollouts are poor-quality placements
- the only full-completion rollout in this probe occurs at `update 3`
  - after that it disappears
  - this supports a "forgetting / instability" interpretation rather than a pure lack-of-capability interpretation

### Geometry note

- final rollout geometry in this completion-bonus probe:
  - `placed_macro_count = 120`
  - `remaining_macro_count = 13`
  - `largest_component_vs_remaining_area_ratio ~= 9.41`
- interpretation:
  - the failure is not simple lack of total free area
  - it is more consistent with late-stage packing / contiguity / ordering quality

### Current conclusion

- keep completion bonus out of the canonical Ariane PPO config for now
- do not continue scaling this specific reward tweak
- the next cheaper lever is more likely a curriculum-style terminal-stage training setup than a larger completion bonus

## 2026-04-13 / 2026-04-14 reverse curriculum v1 and v2 pivot

### Reverse curriculum v1 result

- artifact:
  - `experiments/chip/phase3_reverse_curriculum_probe_10/phase3_reverse_curriculum_probe.json`
- result:
  - `scratch_control_final_hpwl ~= 3.0120M`
  - `reverse_curriculum_final_hpwl ~= 3.1453M`
- behavior-level interpretation:
  - reverse curriculum v1 increased survivability slightly but made full-episode quality worse
  - it learned more of a `tail rescuer` than a better full-episode placer

### Why v1 failed

- the original v1 probe used a **self-generated** prefix pool from a short `base_updates=10` run
- that means two apparent causes were actually one shared bootstrap problem:
  - weak prefix pool
  - weak base run
- `tail_steps=5` also made the curriculum too narrow:
  - reverse history effectively trained on `5-step` suffixes only
  - this produced low-entropy local rescue behavior that did not transfer back to from-scratch eval

### V2 design decision

- do **not** bootstrap the reverse curriculum from its own short base run
- v2 must consume an **external high-quality prefix artifact**
- v2 should keep:
  - `tail_steps = 10` as the first widened horizon
  - prefix filtering by `final_hpwl`
  - a salvage gate:
    - keep only prefixes whose greedy suffix can still complete
- keep scratch control fully independent from reverse curriculum training

### V2 implementation status

- reverse curriculum runner now supports:
  - `prefix_source_path`
  - `prefix_source_hpwl_threshold`
  - external replay via serialized `action_indices`
  - prefix salvage evaluation (`random` + `greedy`)
  - independent scratch / reverse warm-start policies
- exporter now exists for generating replayable external prefix artifacts:
  - `chip/experiments/phase3_macroplacement_ppo.py::export_ariane_prefix_pool`
- this means the **plumbing blocker is closed**

### What still blocks reverse curriculum v2 execution

- there is still **no strong external prefix artifact** in the repo today
- existing Ariane checkpoint JSON artifacts mostly contain aggregate history, not replayable `action_indices`
- next required step is therefore:
  - generate a replayable external prefix pool with `export_ariane_prefix_pool`
  - then rerun reverse curriculum v2 against that external artifact

### Exporter smoke status

- `export_ariane_prefix_pool` now has a real Ariane smoke pass
- smoke artifact:
  - `experiments/chip/phase3_external_prefix_pool_smoke/phase3_ariane_prefix_pool.json`
- smoke config:
  - `num_updates = 1`
  - `candidate_rollouts = 1`
  - `grid_shape = 64x64`
- observed outcome:
  - exporter completed successfully
  - rollout row includes replayable `action_indices`
  - smoke `base_final_rollout.final_hpwl ~= 3.0238M`
- interpretation:
  - the exporter path is real and working on Ariane
  - the remaining blocker is now purely the runtime cost of generating a **strong** external prefix pool, not missing implementation

## 2026-04-14 protocol hardening for reverse curriculum v2

### What changed

- reverse curriculum v2 is no longer just a runner change
- the surrounding data protocol is now explicit in code:
  - accepted prefix-pool criteria
  - multi-seed exported prefix sources
  - separate salvage calibration artifact

### New capabilities

- `chip/experiments/phase3_macroplacement_ppo.py::export_ariane_prefix_pool`
  - now supports:
    - `policy_seeds`
    - `hpwl_threshold`
    - `remaining_macro_range`
    - `geometry_packing_factor`
    - `min_accepted_prefixes`
  - exported rollouts now record:
    - replayable `action_indices`
    - `policy_seed`
    - acceptance flags for reverse curriculum
    - geometry summary when available
  - artifact now reports:
    - `acceptance_criteria`
    - `acceptance_summary`
    - pass / partial / fail status for the pool

- `chip/experiments/phase3_macroplacement_ppo.py::calibrate_ariane_prefix_pool`
  - consumes an external prefix artifact
  - runs prefix-level salvage evaluation
  - records per-prefix `greedy` / `random` suffix outcomes
  - summarizes how many prefixes are actually completable

### Current methodology record

- reverse curriculum v2 should **not** be run directly after export
- first:
  1. export a multi-seed prefix pool
  2. check whether the pool itself passes acceptance
  3. run standalone salvage calibration
  4. only then run the reverse-only v2 probe

### Test / verification status

- focused reverse-curriculum / exporter suite:
  - `17 passed`
- full suite:
  - `python -m pytest tests/engine tests/chip -q`
  - `218 passed`

### Remaining blocker

- reverse curriculum v2 is now blocked only by **runtime cost of producing a strong external pool**
- code and artifact schema are no longer the blocker

### Option-B reuse check on existing canonical64 artifact

- checked:
  - `experiments/chip/phase3_macroplacement_checkpoint_64x64_100/phase3_ppo_ariane_checkpoint.json`
- result:
  - this artifact contains only aggregate history + final rollout
  - it does **not** contain replayable `rollouts`, `action_indices`, or `transitions`
- conclusion:
  - the old canonical `100-update` run cannot be reused directly as the external prefix source for reverse curriculum v2
  - future long runs should preserve replayable rollout data when they may feed downstream curriculum work

### Calibration dry-run on smoke prefix artifact

- smoke calibration artifact:
  - `experiments/chip/phase3_external_prefix_pool_smoke/phase3_ariane_prefix_calibration_smoke.json`
- dry-run config:
  - source = `phase3_external_prefix_pool_smoke/phase3_ariane_prefix_pool.json`
  - `tail_steps = 10`
  - `hpwl_threshold = 2.92`
- result:
  - calibration path completed successfully on real Ariane data
  - `total_candidates = 0`
- interpretation:
  - this is expected because the single smoke rollout had `final_hpwl ~= 3.042M`, so it was filtered out before salvage
  - the important outcome is that the calibration pipeline itself is real and functioning

### Current status summary

- canonical Ariane PPO line:
  - still `Milestone B2 in progress / not passed`
- reverse curriculum:
  - `v1`: negative result, but informative
  - `v2 plumbing`: implemented and tested
  - `v2 experiment`: blocked only on producing the first strong external prefix source

### April 16, 2026: warm-start probe reframed the B2 blocker

- corrected warm-start probe acceptance after HPWL unit normalization bug:
  - source artifact:
    - `experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe.json`
  - recomputed artifacts:
    - `experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe_recomputed_acceptance.json`
    - `experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe_recomputed_full.json`
- key finding:
  - warm-start `update0` completion rate is `0.36`
  - after one PPO update it drops to `0.08`
  - after long canonical PPO it had already fallen to `~0.02`
- interpretation:
  - the main Phase 3 B2 blocker is no longer "how to fabricate better prefixes"
  - it is "how to stop PPO from destroying a BC warm-start policy that already works materially better than later PPO checkpoints"
- methodological consequence:
  - reverse curriculum is now demoted from primary path to optional fallback
  - the new primary experiment path is `warm-start + KL-anchored PPO`

### April 16, 2026: KL-anchored PPO wiring landed

- `chip/experiments/phase3_ppo.py`
  - `run_single_ppo_update()` now supports:
    - `reference_policy`
    - `kl_anchor_beta`
  - per-update metrics now record:
    - `kl_anchor_loss`
    - `kl_anchor_beta`
- `chip/experiments/phase3_macroplacement_ppo.py`
  - canonical Ariane checkpoint config now routes to:
    - `build_ariane_warm_kl_anchor_training_config()`
    - default `kl_anchor_beta = 5.0`
  - warm-start training entry points now clone a frozen reference policy and pass it into PPO when `kl_anchor_beta > 0`:
    - `run_ariane_ppo_checkpoint`
    - `run_ariane_termination_audit`
    - `export_ariane_warm_start_probe`
    - `export_ariane_prefix_pool`
    - reverse-curriculum probe warm-start branches

### Verification status

- focused KL-anchor tests:
  - `python -m pytest tests/chip/test_phase3_ppo.py tests/chip/test_phase3_macroplacement_ppo.py -q -k "kl_anchor or run_ariane_ppo_checkpoint_records_gap_progress or export_ariane_warm_start_probe_records_update0_update1_and_salvage or run_single_ppo_update_returns_finite_metrics"`
  - `4 passed`
- widened Phase 3 PPO suite:
  - `python -m pytest tests/chip/test_phase3_ppo.py tests/chip/test_phase3_macroplacement_ppo.py -q`
  - `37 passed`
- full engine + chip suite:
  - `python -m pytest tests/engine tests/chip -q`
  - `223 passed`

### Immediate next step

- run a real Ariane checkpoint/probe with:
  - BC warm-start
  - `kl_anchor_beta = 5.0`
  - existing `1e-4` / high-entropy warm config
- success criterion for next diagnosis slice:
  - update-1 completion should no longer collapse from `0.36` toward `0.08`
  - if KL anchor preserves warm-start completion materially, reverse curriculum stays optional
  - if KL anchor still collapses, then the next formulation diagnosis should move to reward / objective structure rather than prefix sourcing

### April 16, 2026: first real KL-anchor probe ran, but anchor is inert on update 1

- smoke artifact:
  - `experiments/chip/phase3_warm_start_probe_kl_anchor_smoke/phase3_ariane_warm_start_probe.json`
  - `update0_completion_rate = 0.0`
  - `update1_completion_rate = 0.0`
  - `update0_median_hpwl = 2.8895M`
  - `update1_median_hpwl = 3.0041M`
- wider probe artifact:
  - `experiments/chip/phase3_warm_start_probe_kl_anchor_10x10/phase3_ariane_warm_start_probe.json`
  - `update0_completion_rate = 0.0`
  - `update1_completion_rate = 0.0`
  - `update0_median_hpwl = 2.8670M`
  - `update1_median_hpwl = 2.9932M`
  - `update0_p10_hpwl = 2.7865M`
  - `update1_p10_hpwl = 2.8549M`
  - `update0_accepted = 0`
  - `update1_accepted = 0`
- critical diagnostic:
  - `update1` training history recorded:
    - `kl_anchor_beta = 5.0`
    - `kl_anchor_loss = 0.0`
- interpretation:
  - the current KL-anchor implementation does reach the real Ariane loop
  - but it is **inactive on the first PPO step**, because the reference policy and trainable policy are identical before update 1
  - therefore the anchor does not prevent the very first destructive move away from BC warm-start
- methodological consequence:
  - the current "single-step KL penalty to frozen reference" is not sufficient to solve the observed warm-start collapse
  - next diagnosis should focus on objective / optimizer structure that can constrain the first update itself:
    - multi-epoch PPO updates where KL becomes active after the first substep
    - explicit distillation / behavior-cloning retention loss on teacher states during PPO
    - smaller or trust-region-like first update schedule

### April 16, 2026: substep-level diagnosis completed

- new diagnostic support landed in:
  - `chip/experiments/phase3_ppo.py`
    - `run_single_ppo_update(..., policy_epochs, diagnostic_eval_seeds)` now records:
      - `pre_update_diagnostic`
      - `substep_history`
  - `chip/experiments/phase3_macroplacement_ppo.py`
    - `run_ariane_warm_start_substep_diagnostic(...)`
- real diagnostic artifact on generic held-out seeds:
  - `experiments/chip/phase3_warm_start_substep_diagnostic_5x5/phase3_ariane_warm_start_substep_diagnostic.json`
  - pre-update:
    - `heldout_mean_hpwl = 2.8449M`
    - `heldout_log_prob_delta_mean ~= 0`
    - `update_batch_kl_to_reference = 0`
  - after first optimizer substep (`substep_index = 0`):
    - `heldout_mean_hpwl = 2.8716M`
    - `heldout_log_prob_delta_mean = -0.244`
    - `update_batch_kl_to_reference = inf`
  - interpretation:
    - destructive drift begins on the **first optimizer substep**
    - this supports the hypothesis that "multi-epoch alone" is not enough if substep 0 is already damaging
- follow-up diagnostic on seeds that had appeared as completion-positive in the older warm-start probe:
  - `experiments/chip/phase3_warm_start_substep_diagnostic_completed_seeds/phase3_ariane_warm_start_substep_diagnostic.json`
  - direct replay check:
    - the old completion-positive seeds (`2, 3, 4, 5, 7`) no longer reproduce as completed episodes under the current rerun
  - implication:
    - completion on stochastic warm-start rollouts is not currently stable enough to be the primary diagnostic target
    - the most trustworthy diagnosis signal in this slice is:
      - teacher-state `log_prob_delta`
      - held-out HPWL drift
      - update-batch KL drift

### Updated B2 diagnosis

- answer to the "Option 1 vs Option 2" question:
  - the new evidence supports:
    - **substep-0 damage is real**
    - therefore "just let KL turn on in later substeps" is probably insufficient as the only fix
- preferred next stabilization slice:
  - keep the new substep diagnostic harness
  - test a **substep-0-active constraint**, e.g.:
    - behavior-cloning retention loss on teacher states during PPO
    - or first-update LR warmup as the cheapest mechanical control
  - do **not** prioritize reverse curriculum before this is resolved

### April 16, 2026: current-code warm-start-only baseline no longer supports the old 0.36 anchor

- reproducibility check artifact:
  - `experiments/chip/phase3_warm_start_only_probe_20/phase3_ariane_warm_start_only_probe.json`
- current rerun result:
  - `N = 20`
  - `completed_count = 0`
  - `completion_rate = 0.0`
  - `min_hpwl = 2.7825M`
  - `median_hpwl = 2.8818M`
  - `mean_hpwl = 2.9073M`
- implication:
  - the old historical `warm-start completion_rate = 0.36` should no longer be used as the operative anchor for current-code success criteria
  - for the present code state, the most reproducible warm-start capability signal is:
    - low stochastic HPWL distribution
    - not completion rate
- consequence for next diagnosis slice:
  - success/failure should be framed around preserving warm-start HPWL / teacher-state log-prob geometry first
  - any completion-based gate needs to be re-established from new artifacts, not inherited from older probes

### April 16, 2026: epoch-1 batch diagnostic clarifies the substep-0 failure shape

- artifact:
  - `experiments/chip/phase3_warm_start_substep_diagnostic_epoch1/phase3_ariane_warm_start_substep_diagnostic.json`
- pre-step batch diagnostic:
  - `return_summary`:
    - `count = 105`
    - `min = -166157`
    - `mean = -27463`
    - `max = 267031`
  - `old_value_summary`:
    - `min = 49.5`
    - `mean = 52.5`
    - `max = 58.6`
  - `raw_advantage_summary` is effectively identical to returns at this scale
  - `normalized_advantage_summary` is *not* extremely heavy-tailed after normalization:
    - `min = -1.05`
    - `max = 2.23`
  - `value_return_correlation = 0.85`
- substep-0 diagnostic:
  - `update_batch_kl_to_reference = inf`
  - `global_grad_norm = 82.06`
  - `policy_head_grad_norm = 21.20`
  - `value_head_grad_norm = 0.0`
  - held-out drift still appears immediately:
    - `heldout_mean_hpwl: 2.8449M -> 2.8716M`
    - `heldout_log_prob_delta_mean: ~0 -> -0.244`
- interpretation:
  - the critic path is currently not helping at all in this configuration:
    - `value_coef = 0.0`
    - `value_head_grad_norm = 0.0`
    - old values stay on a tiny `~50` scale while returns are `O(1e5)`
  - this supports the diagnosis that current PPO is effectively operating as a badly-baselined policy-gradient update
  - however, the normalized advantages are not singularly extreme after normalization, so "one absurd normalized outlier" is **not** the full explanation by itself
  - next single-variable intervention should target the absent critic signal before broader retention-style constraints

### Reproducibility invariant

- completion-rate observations must always be tagged with:
  - code version / branch state
  - seed management scheme
  - artifact path
- historical completion numbers without those tags are not valid operative anchors for current Phase 3 decisions

### April 16, 2026: offline value pretraining was tested as a single-variable intervention and failed the necessary gate

- strict artifact:
  - `experiments/chip/phase3_critic_prefit_diagnostic_v1_strict/phase3_ariane_critic_prefit_diagnostic.json`
- experiment design:
  - prefit on warm-start rollouts from seeds `(0, 1, 2, 3)`
  - `prefit_epochs = 20`
  - then run exactly one PPO optimizer substep with the usual warm-start PPO config
  - follow-up 10-update training is now skipped automatically when the necessary gate fails
- prefit result:
  - `initial_loss = 1.8065e10`
  - `final_loss = 1.7928e10`
  - only a very small loss reduction
- post-prefit batch diagnostic:
  - `old_value_summary` moved from `~ +50` to about `-1642`
  - returns remain `O(1e5)`
  - `value_return_correlation = -0.718`
- substep-0 result after prefit:
  - `update_batch_kl_to_reference = inf`
  - `heldout_log_prob_delta_mean = -0.244`
  - `heldout_mean_hpwl = 2.8716M`
  - `value_head_grad_norm = 0.0`
- gate outcome:
  - necessary:
    - `finite_kl = false`
    - `heldout_log_prob_delta_within_band = false`
    - `pass = false`
  - sufficient:
    - not evaluated in the strict workflow because necessary gate failed
- conclusion:
  - **offline value-head pretraining alone is not enough**
  - the next most justified single-variable intervention is no longer "more offline prefit"
  - it is to test an **online critic warmup** / nonzero critic training path before the first policy update

### April 16, 2026: Step-1 critic diagnosis refined the failure mode further

- architecture check:
  - `engine/ppo_policy.py`
  - `value_head = Linear(hidden_dim*3 -> hidden_dim) + ReLU + Linear(hidden_dim -> 1)`
  - there is **no tanh or bounded output activation** on the final value head
- implication:
  - the strongest version of the "bounded/saturating value head output" hypothesis is not supported
  - however, raw-target conditioning is still a live issue because outputs initialize near `O(1e1-1e3)` while returns are `O(1e5)`
- return-shape check on current warm-start rollouts (seeds `0..3`):
  - returns are broad, sign-changing, and not obviously simple/bimodal
  - example seed-0 quantiles:
    - `p10 = -148366`
    - `p50 = -95392`
    - `p90 = 178980`
  - similar wide sign-crossing behavior appears on seeds `1..3`
- simple feature correlations over those warm-start rollouts:
  - `episode_return_vs_final_hpwl = -1.0` on the 4-sample probe
  - `episode_return_vs_steps ~= -0.35`
  - interpretation:
    - there is at least some simple rollout-level signal in the task
    - so the critic failure is not obviously "no signal exists at all"
- updated interpretation:
  - current evidence points more toward **target-scale / optimization conditioning failure** than "bounded value-head architecture failure"
  - this weakens the case for jumping straight to online critic warmup as the next experiment
  - the more targeted next single-variable test is now:
    - offline critic prefit with **transformed/scaled targets** (e.g. z-score or signed-log returns)
    - then re-run the same substep-0 necessary gate

### April 16, 2026: training-fair-shot alone did not fix critic prefit, but z-score target scaling did

- artifact:
  - `experiments/chip/phase3_critic_prefit_comparison_v1/phase3_ariane_critic_prefit_comparison.json`
- experiment design:
  - same warm-start rollout pool for all three conditions
  - conditions:
    - `raw_baseline`: `epochs = 20`, `lr = 1e-3`, `lr_warmup_epochs = 0`
    - `raw_fair_shot`: `epochs = 200`, `lr = 1e-3`, `lr_warmup_epochs = 20`
    - `zscore_fair_shot`: `epochs = 200`, `lr = 1e-3`, `lr_warmup_epochs = 20`, target = z-scored return
- simple signal diagnostic from the same rollout pool:
  - `transition_count = 525`
  - `return_to_go_vs_step_index = -0.655`
  - `return_to_go_vs_remaining_macros = +0.655`
  - `return_to_go_vs_current_hpwl = +0.710`
  - `episode_return_vs_final_hpwl = -1.0`
- condition outcomes:
  - `raw_baseline`:
    - `final_value_return_correlation = -0.555`
    - `final_loss = 2.1165e10`
  - `raw_fair_shot`:
    - `final_value_return_correlation = -0.556`
    - `final_loss = 1.9480e10`
  - `zscore_fair_shot`:
    - `final_value_return_correlation = +0.853`
    - `final_loss = 0.274`
- gate result:
  - `raw_baseline`: fail
  - `raw_fair_shot`: fail
  - `zscore_fair_shot`: pass
- interpretation:
  - giving raw-target prefit a "fair shot" with more epochs and LR warmup **did not** rescue the critic
  - the failure is therefore not just "prefit was undertrained"
  - the decisive change was **target scaling**, not longer raw-target optimization
  - current best explanation is:
    - the state representation does carry usable value signal
    - but raw return scale/conditioning makes the critic learn an unusable mapping
- implication for next Phase 3 B2 slice:
  - do **not** jump to online critic warmup on raw targets
  - first rerun the substep-0 necessary gate with a **z-score-prefit critic**
  - only if that still fails should the next layer of stabilization (LR warmup, grad clipping, BC retention, etc.) be considered
