# Phase 3 Fragile-Decision Probe And Closure Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a pre-registered Ariane fragile-decision importance probe that can unblock Phase 3 closure, while in parallel building the minimum real-benchmark SA table skeleton needed for the final comparison harness.

**Architecture:** Track A uses a fixed warm-start `pi_init` trace plus paired single-step counterfactual suffix rollouts to estimate whether fragile flips are negligible, concentrated, or init-rooted. Track B generalizes MacroPlacement benchmark resolution beyond Ariane and emits an SA-only HPWL table for three real macro-placement benchmarks so closure work can proceed regardless of PPO outcome.

**Tech Stack:** PyTorch PPO policy, `PlacementEnv` / `ChipRLAdapter`, MacroPlacement assets in `tmp/MacroPlacement`, existing SA solver, JSON artifacts, pytest.

---

## Prereg Snapshot

### Investigation Goal

Track A answers one question:

> Do the Ariane fragile flips identified by the argmax diagnostics materially change placement quality, or are they mostly low-impact tie-boundary perturbations?

Track B answers one infrastructure question:

> Can the Phase 3 comparison harness already support three real macro-placement benchmarks for SA rows without waiting for PPO closure?

### Facts Locked Before Implementation

- Baseline first-substep diagnostic on Ariane:
  - `agreement = 0.6165`
  - `flip_count = 51 / 133`
  - flips are enriched in `high entropy / low margin`
- `alpha/10` null-reference diagnostic on Ariane:
  - `agreement = 0.9173`
  - `flip_count = 11 / 133`
  - all `11` flips are a strict subset of baseline flips
  - `8 / 11` point in the same post-update argmax direction as baseline
- HPWL micro-diagnostic is closed at the current `5`-seed protocol:
  - `5`-seed zero-update SEM is `~1.57%`
  - disjoint seed-panel zero-update shift is `+3.59%`
- Runtime probes already show full exhaustive counterfactual evaluation is not feasible under a `3`-day wall-clock cap:
  - full Ariane stochastic rollout from step `0`: `~150.5s`
  - suffix rollout from a replayed later prefix (`~step 80`): `~90.7s`

### Track A Protocol

#### Reference Trace

- Use a **fixed deterministic warm-start `pi_init` trace** on Ariane.
- Keep the reference trace fixed for the entire probe.
- Counterfactual rollouts replay the same prefix through step `k - 1`, replace only step `k`, and then continue with stochastic warm-start suffix rollout.

#### Counterfactual Mode

- Primary mode: **single-step replacement, state-consistent**
  - replay fixed prefix
  - force one replacement action at step `k`
  - continue suffix with warm-start `pi_init`
- Compare paired suffix rollouts under the **same suffix seed**:
  - original action at step `k`
  - counterfactual replacement action at step `k`

#### Evaluated Step Set

Full `51`-flip evaluation is excluded by budget. The evaluated subset is:

- `11` overlap flips:
  - all steps shared by `alpha/10` and baseline first-substep drift
- `11` baseline-unique flips:
  - selected from the remaining baseline-only flips by **lowest pre-update margin first**

This yields a fixed `22`-step priority subset.

Rationale:

- preserves the fragile-core overlap cohort intact
- keeps a balanced overlap-vs-unique comparison for init-rooted analysis
- fits within the `3`-day wall-clock cap with `25` paired suffix seeds

#### Replacement Actions

- For all `22` selected steps:
  - evaluate **baseline post-substep0 argmax** as the primary replacement action
- For the `11` overlap steps only:
  - also evaluate **`alpha/10` post-substep0 argmax** as a secondary calibration panel

#### Seed Budget

- Use `25` suffix seeds per evaluated step.
- Use the **same suffix-seed panel** for original and counterfactual branches at each step.
- No deterministic evaluator swap is allowed in this prereg; keep the existing evaluation semantics and reduce variance via seed count only.

### Track A Decision Thresholds

#### Materiality Scale

Anchor per-step importance to the Ariane greedy-vs-masked-random gap:

- `gap = 3.2218M - 1.6826M = 1.5392M`
- `1% of gap = 15,392 HPWL`
- `0.2% of gap = 3,078 HPWL`

Interpretation:

- **material** per-step effect: `|delta_hpwl| > 15,392`
- **negligible** per-step effect: `|delta_hpwl| < 3,078`
- otherwise: **modest**

#### Concentration

Concentration is computed on the evaluated-step panel using absolute paired mean HPWL deltas.

- **strong concentration**:
  - top `20%` of evaluated steps contribute `> 80%` of total `|delta|`
- **moderate concentration**:
  - top `20%` contribute `50%` to `80%`
- **diffuse**:
  - top `20%` contribute `< 50%`

#### Init-Rooted Null

Primary comparison:

- overlap cohort = `11` shared flips
- unique cohort = `11` lowest-margin baseline-only flips

Null:

- `median |delta| (overlap) == median |delta| (unique)`

Decision rule:

- preferred: Mann-Whitney U test on per-step `|delta|`
- fallback if test plumbing becomes disproportionate:
  - `median_ratio = median |delta| (overlap) / median |delta| (unique)`
  - treat `[0.5, 2.0]` as same order-of-magnitude evidence for init-rooted fragility

### Track A Outcome Interpretation

- **Impact small**:
  - median `|delta|` negligible or modest
  - no strong concentration
  - closure interpretation: PPO stabilization is not the main blocker
- **Impact concentrated**:
  - strong concentration with a small number of material steps
  - closure interpretation: bottleneck is sparse critical decisions, not uniform policy failure
- **Impact init-rooted**:
  - overlap and unique cohorts remain same-order-of-magnitude
  - closure interpretation: fragility lives in the warm-start policy itself, not only in baseline PPO aggressiveness

### Track B Scope

Track B is intentionally narrower than the final unified table:

- add a **generic real-benchmark resolver** that supports at least:
  - `ariane133`
  - `bp_quad`
  - `nvdla`
- emit a **SA-only HPWL table skeleton** for those three benchmarks
- do **not** block on:
  - analytical rows
  - OpenROAD rows
  - routed / timing / runtime metrics

### Wall-Clock Cap

- Hard cap: `3` days wall-clock for Track A probe work.
- If the full pre-registered `22`-step panel threatens the cap in practice:
  - stop at the largest completed prefix of the pre-registered priority order
  - do not expand the step set or change seed count mid-run

## File Structure

### Track A

- Modify: `chip/experiments/phase3_macroplacement_ppo.py`
  - add Ariane fixed-trace counterfactual probe helpers and runner
- Modify: `chip/experiments/phase3_ppo.py`
  - only if a small generic replay helper is clearly reusable
- Test: `tests/chip/test_phase3_macroplacement_ppo.py`
  - probe cohort selection
  - paired counterfactual summary
  - concentration / materiality / init-rooted summary rules

### Track B

- Modify: `chip/experiments/phase3_macroplacement_benchmarks.py`
  - generic benchmark resolver for MacroPlacement real benchmarks
- Modify: `chip/experiments/phase3_roadmap_eval.py`
  - or add a small real-benchmark SA-table helper if extending in place stays clean
- Test: `tests/chip/test_phase3_macroplacement_benchmarks.py`
  - generic resolver coverage for `bp_quad` and `nvdla`
- Test: `tests/chip/test_phase3_roadmap_eval.py`
  - SA-only real-benchmark skeleton row presence for `3` benchmarks

### Documentation / Artifacts

- Create: `docs/superpowers/plans/2026-04-17-phase3-fragile-decision-probe-and-closure-plan.md`
- Modify: `docs/superpowers/task-memory.md`
- Artifact directories:
  - `experiments/chip/phase3_fragile_decision_probe_v1/`
  - `experiments/chip/phase3_real_benchmark_sa_table_v1/`

### Task 1: Track A Probe Cohort Selection And Summary Primitives

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_macroplacement_ppo.py`
- Test: `C:\Users\14574\Chip\tests\chip\test_phase3_macroplacement_ppo.py`

- [ ] **Step 1: Write the failing tests for priority subset selection and delta summarization**

```python
def test_select_fragile_decision_priority_subset_keeps_overlap_and_low_margin_unique():
    flips = {
        "baseline_flipped_step_indices": [1, 2, 3, 4, 5, 6],
        "alpha10_flipped_step_indices": [1, 3],
        "pre_update_margin_by_step": {
            1: 0.01,
            2: 0.04,
            3: 0.02,
            4: 0.10,
            5: 0.03,
            6: 0.20,
        },
    }

    summary = _select_fragile_decision_priority_subset(
        baseline_flipped_step_indices=flips["baseline_flipped_step_indices"],
        alpha10_flipped_step_indices=flips["alpha10_flipped_step_indices"],
        pre_update_margin_by_step=flips["pre_update_margin_by_step"],
        unique_budget=2,
    )

    assert summary["overlap_steps"] == [1, 3]
    assert summary["unique_steps"] == [5, 2]
    assert summary["evaluated_steps"] == [1, 3, 5, 2]


def test_summarize_fragile_decision_probe_applies_materiality_and_concentration_thresholds():
    per_step = [
        {"step_index": 1, "mean_hpwl_delta": 20000.0, "cohort": "overlap"},
        {"step_index": 2, "mean_hpwl_delta": 18000.0, "cohort": "overlap"},
        {"step_index": 3, "mean_hpwl_delta": 1000.0, "cohort": "unique"},
        {"step_index": 4, "mean_hpwl_delta": 500.0, "cohort": "unique"},
    ]

    summary = _summarize_fragile_decision_probe(
        per_step_results=per_step,
        greedy_random_gap_hpwl=1_539_200.0,
    )

    assert summary["material_step_count"] == 2
    assert summary["concentration_band"] == "strong"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k "fragile_decision_priority_subset or fragile_decision_probe_applies" -q`

Expected: FAIL with missing helper function errors.

- [ ] **Step 3: Implement the minimal subset-selection and summary helpers**

```python
def _select_fragile_decision_priority_subset(
    *,
    baseline_flipped_step_indices: list[int],
    alpha10_flipped_step_indices: list[int],
    pre_update_margin_by_step: dict[int, float],
    unique_budget: int = 11,
) -> dict[str, object]:
    overlap = sorted(set(alpha10_flipped_step_indices) & set(baseline_flipped_step_indices))
    unique_candidates = [
        step
        for step in baseline_flipped_step_indices
        if step not in set(overlap)
    ]
    unique_sorted = sorted(unique_candidates, key=lambda step: (pre_update_margin_by_step[step], step))
    unique = unique_sorted[:unique_budget]
    return {
        "overlap_steps": overlap,
        "unique_steps": unique,
        "evaluated_steps": [*overlap, *unique],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k "fragile_decision_priority_subset or fragile_decision_probe_applies" -q`

Expected: PASS

### Task 2: Track A Fixed-Trace Single-Step Counterfactual Runner

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_macroplacement_ppo.py`
- Test: `C:\Users\14574\Chip\tests\chip\test_phase3_macroplacement_ppo.py`

- [ ] **Step 1: Write the failing runner test for paired original-vs-counterfactual suffix evaluation**

```python
def test_run_ariane_fragile_decision_probe_records_paired_suffix_results(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(module, "_collect_fixed_reference_trace", fake_reference_trace)
    monkeypatch.setattr(module, "_run_single_step_counterfactual_panel", fake_probe_panel)

    artifact = module.run_ariane_fragile_decision_probe(output_path=tmp_path / "probe.json")

    assert artifact["probe_config"]["suffix_seed_count"] == 25
    assert artifact["priority_subset"]["overlap_count"] == 11
    assert artifact["priority_subset"]["unique_count"] == 11
    assert "baseline_panel" in artifact["panels"]
    assert "alpha10_overlap_panel" in artifact["panels"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k fragile_decision_probe_records_paired_suffix_results -q`

Expected: FAIL with missing runner/helper errors.

- [ ] **Step 3: Implement the minimal runner**

```python
def run_ariane_fragile_decision_probe(*, output_path: Path) -> dict[str, object]:
    # 1. load warm-start / baseline / alpha10 policies
    # 2. collect fixed deterministic reference trace
    # 3. derive priority subset from existing argmax diagnostics
    # 4. run paired single-step suffix panels
    # 5. summarize materiality / concentration / init-rooted interpretation
    # 6. write JSON artifact
    ...
```

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k "fragile_decision_probe or argmax_agreement" -q`

Expected: PASS

### Task 3: Track B Generic Real-Benchmark Resolver

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_macroplacement_benchmarks.py`
- Test: `C:\Users\14574\Chip\tests\chip\test_phase3_macroplacement_benchmarks.py`

- [ ] **Step 1: Write the failing tests for generic benchmark resolution**

```python
def test_resolve_macroplacement_benchmark_supports_bp_quad():
    spec = resolve_macroplacement_benchmark(
        repo_root=macroplacement_repo_root(),
        variant="bp_quad",
        platform="NanGate45",
    )
    assert spec.variant == "bp_quad"
    assert spec.sdc_path.name == "bsg_chip.sdc"


def test_resolve_macroplacement_benchmark_supports_nvdla():
    spec = resolve_macroplacement_benchmark(
        repo_root=macroplacement_repo_root(),
        variant="nvdla",
        platform="NanGate45",
    )
    assert spec.variant == "nvdla"
    assert spec.sdc_path.name == "nvdla.sdc"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_benchmarks.py -k "bp_quad or nvdla" -q`

Expected: FAIL with missing generic resolver.

- [ ] **Step 3: Implement the generic resolver and keep Ariane wrapper compatibility**

```python
def resolve_macroplacement_benchmark(
    *,
    repo_root: Path | None = None,
    variant: str,
    platform: str = "NanGate45",
) -> MacroPlacementBenchmarkSpec:
    ...


def resolve_ariane_benchmark(...):
    return resolve_macroplacement_benchmark(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/chip/test_phase3_macroplacement_benchmarks.py -q`

Expected: PASS

### Task 4: Track B SA-Only Real-Benchmark Table Skeleton

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_roadmap_eval.py`
- Test: `C:\Users\14574\Chip\tests\chip\test_phase3_roadmap_eval.py`

- [ ] **Step 1: Write the failing test for a 3-benchmark SA skeleton**

```python
def test_build_real_macroplacement_sa_table_contains_three_benchmarks():
    report = build_phase3_real_macroplacement_sa_table()

    assert report["benchmarks"] == ("ariane133_ng45", "bp_quad_ng45", "nvdla_ng45")
    assert "sa" in report["method_rows"]
    assert set(report["method_rows"]["sa"]) == set(report["benchmarks"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/chip/test_phase3_roadmap_eval.py -k real_macroplacement_sa_table -q`

Expected: FAIL with missing helper.

- [ ] **Step 3: Implement the minimal SA-only real-benchmark table helper**

```python
def build_phase3_real_macroplacement_sa_table() -> dict[str, object]:
    variants = ("ariane133", "bp_quad", "nvdla")
    ...
    return {
        "benchmarks": (...),
        "method_rows": {"sa": ...},
        "status": "skeleton",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/chip/test_phase3_roadmap_eval.py -q`

Expected: PASS

### Task 5: Verification, Artifact Emission, And Memory Update

**Files:**
- Modify: `C:\Users\14574\Chip\docs\superpowers\task-memory.md`

- [ ] **Step 1: Run the focused test suites**

Run:

```bash
python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -q
python -m pytest tests/chip/test_phase3_macroplacement_benchmarks.py -q
python -m pytest tests/chip/test_phase3_roadmap_eval.py -q
```

Expected: PASS

- [ ] **Step 2: Run broader verification**

Run:

```bash
python -m pytest tests/engine tests/chip -q
```

Expected: PASS

- [ ] **Step 3: Update task memory**

Add:

```md
## 2026-04-17 Phase 3 Fragile-Decision Probe / Closure Skeleton

- prereg locked with fixed `pi_init` trace, `22`-step priority subset, and `25` paired suffix seeds
- Track B skeleton started with real-benchmark SA rows for Ariane / bp_quad / nvdla
```

## Self-Review

- Spec coverage:
  - Track A prereg, runtime-driven subseting, materiality thresholds, concentration bands, and init-rooted comparison are all explicitly covered.
  - Track B minimal SA-only real-benchmark skeleton is explicitly covered.
- Placeholder scan:
  - no `TODO` / `TBD` placeholders remain in execution tasks
- Scope check:
  - deliberately limited to Direction 3 probe + SA-only closure skeleton
  - does not attempt full unified table in this slice

## Execution Handoff

The user already explicitly chose execution in this session (`"开干"`), so proceed inline rather than waiting for a separate execution-choice prompt.
