# Phase 3 Macroplacement Ariane-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch Phase 3 onto a true macro-placement benchmark path by proving out `Ariane` first.

**Architecture:** Keep existing Phase 1/2 benchmark infrastructure intact, add a new macro-placement benchmark asset path and an Ariane-first probe/eval harness, and make the next Phase 3 decision from measured legality and OpenROAD costs rather than assumption.

**Tech Stack:** Python, pytest, existing LEF/DEF parser, OpenROAD Docker flow, JSON artifacts

---

## File Structure

- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_benchmarks.py`
  - Ariane asset discovery/fetch helpers and benchmark metadata
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_probe.py`
  - Ariane OpenROAD eval + legality/per-step feasibility probe
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\tests\chip\test_phase3_macroplacement_benchmarks.py`
  - benchmark manifest and asset path tests
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\tests\chip\test_phase3_macroplacement_probe.py`
  - probe artifact structure tests

### Task 1: Ariane Benchmark Manifest

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_benchmarks.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\tests\chip\test_phase3_macroplacement_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase3_macroplacement_benchmarks import build_macroplacement_manifest


def test_build_macroplacement_manifest_contains_ariane():
    manifest = build_macroplacement_manifest()
    assert "ariane" in manifest
    assert "asset_root" in manifest["ariane"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_macroplacement_benchmarks.py::test_build_macroplacement_manifest_contains_ariane -q`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

```python
def build_macroplacement_manifest():
    return {"ariane": {"asset_root": None}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_macroplacement_benchmarks.py::test_build_macroplacement_manifest_contains_ariane -q`
Expected: PASS

### Task 2: Ariane Probe Artifact

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_probe.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\tests\chip\test_phase3_macroplacement_probe.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase3_macroplacement_probe import build_probe_report


def test_build_probe_report_contains_core_sections():
    report = build_probe_report(
        benchmark_name="ariane",
        movable_macro_count=42,
        legality_check_seconds=0.01,
        action_mask_seconds=0.5,
        openroad_metrics={"wns": -0.1},
    )
    assert report["benchmark"] == "ariane"
    assert "feasibility_status" in report
    assert "openroad_metrics" in report
```
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_macroplacement_probe.py::test_build_probe_report_contains_core_sections -q`
Expected: FAIL with import error

- [ ] **Step 3: Write minimal implementation**

```python
def build_probe_report(*, benchmark_name, movable_macro_count, legality_check_seconds, action_mask_seconds, openroad_metrics):
    return {
        "benchmark": benchmark_name,
        "movable_macro_count": movable_macro_count,
        "legality_check_seconds": legality_check_seconds,
        "action_mask_seconds": action_mask_seconds,
        "openroad_metrics": openroad_metrics,
        "feasibility_status": "unknown",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_macroplacement_probe.py::test_build_probe_report_contains_core_sections -q`
Expected: PASS

### Task 3: Ariane Asset And Eval Bring-Up

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_benchmarks.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-macroplacement\chip\experiments\phase3_macroplacement_probe.py`

- [ ] **Step 1: Locate or fetch Ariane assets**

Run:

```powershell
python -m chip.experiments.phase3_macroplacement_benchmarks
```

Expected: writes a manifest or prints Ariane asset paths

- [ ] **Step 2: Run pure Ariane OpenROAD eval**

Run:

```powershell
python -m chip.experiments.phase3_macroplacement_probe --benchmark ariane --mode eval
```

Expected: emits `HPWL / routed WL / WNS / TNS / DRC / runtime`

- [ ] **Step 3: Run Ariane legality/action-mask probe**

Run:

```powershell
python -m chip.experiments.phase3_macroplacement_probe --benchmark ariane --mode probe
```

Expected: emits movable macro count, single legality-check latency, and full `build_action_mask()` latency

### Task 4: Final Verification

**Files:**
- Verify only the new Ariane-first files and their artifacts

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/chip/test_phase3_macroplacement_benchmarks.py tests/chip/test_phase3_macroplacement_probe.py -q`
Expected: PASS

- [ ] **Step 2: Run broader suite**

Run: `pytest tests/engine tests/chip -q`
Expected: PASS

- [ ] **Step 3: Record the decision**

Write one artifact that states either:

- `ariane_full_design_ppo_feasible`
- `ariane_full_design_ppo_blocked_by_legality_performance`

## Self-Review

Spec coverage:

- Ariane-first scope: Tasks 1-3
- pure eval sanity: Task 3
- legality/action-mask probe: Task 3
- explicit feasibility decision: Task 4

Placeholder scan:

- no `TBD`
- all commands and files are concrete

Type consistency:

- benchmark key is `ariane`
- decision field is `feasibility_status`
