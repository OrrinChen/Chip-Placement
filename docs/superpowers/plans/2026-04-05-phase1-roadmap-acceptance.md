# Phase 1 Roadmap Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Phase 1 up to the roadmap's original acceptance standard by selecting three public ISPD benchmarks, injecting SA placements into OpenROAD with `place_inst`, evaluating built-in and SA placements through one shared flow template, and emitting a reproducible comparison table.

**Architecture:** Keep the engine untouched and build a narrow chip-domain acceptance overlay. Add a benchmark catalog, a Tcl placement-injection generator, a single-template OpenROAD evaluator, and an acceptance runner that produces one comparison artifact under `experiments/chip/`.

**Tech Stack:** Python 3.11, pytest, pathlib, json, dataclasses, subprocess, Dockerized OpenROAD, Tcl script generation

---

## File Map

- Create: `chip/ispd_benchmarks.py`
- Create: `chip/openroad_eval.py`
- Create: `chip/experiments/phase1_acceptance.py`
- Modify: `chip/experiments/baseline.py`
- Modify: `chip/openroad_bridge.py`
- Create: `tests/chip/test_ispd_benchmarks.py`
- Create: `tests/chip/test_openroad_eval.py`
- Create: `tests/chip/test_phase1_acceptance.py`
- Modify: `tests/chip/test_openroad_bridge.py`

### Task 1: Add ISPD Benchmark Catalog

**Files:**
- Create: `chip/ispd_benchmarks.py`
- Test: `tests/chip/test_ispd_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS


def test_acceptance_benchmarks_cover_three_scale_buckets():
    assert len(ACCEPTANCE_BENCHMARKS) == 3
    assert [benchmark.scale_bucket for benchmark in ACCEPTANCE_BENCHMARKS] == ["small", "medium", "large"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_ispd_benchmarks.py::test_acceptance_benchmarks_cover_three_scale_buckets -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chip.ispd_benchmarks'`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ISPDBenchmark:
    name: str
    source_family: str
    scale_bucket: str
    artifact_dir: str
    compatibility_status: str = "candidate"


ACCEPTANCE_BENCHMARKS = [
    ISPDBenchmark("ibm01", "ISPD 1998", "small", "ibm01"),
    ISPDBenchmark("adaptec1", "ISPD 2005", "medium", "adaptec1"),
    ISPDBenchmark("bigblue1", "ISPD 2005", "large", "bigblue1"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_ispd_benchmarks.py::test_acceptance_benchmarks_cover_three_scale_buckets -v`
Expected: PASS

- [ ] **Step 5: Add a failing path-resolution test**

```python
from pathlib import Path

from chip.ispd_benchmarks import resolve_artifact_paths


def test_resolve_artifact_paths_points_to_benchmark_bundle(tmp_path):
    benchmark_root = tmp_path / "benchmarks"

    paths = resolve_artifact_paths(benchmark_root, "ibm01")

    assert paths["lef_path"] == benchmark_root / "ibm01" / "design.lef"
    assert paths["def_path"] == benchmark_root / "ibm01" / "design.def"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_ispd_benchmarks.py::test_resolve_artifact_paths_points_to_benchmark_bundle -v`
Expected: FAIL with `ImportError` or `AttributeError` for missing `resolve_artifact_paths`

- [ ] **Step 7: Implement the minimal path resolver**

```python
from pathlib import Path


def resolve_artifact_paths(benchmark_root: Path, benchmark_name: str) -> dict[str, Path]:
    root = Path(benchmark_root) / benchmark_name
    return {
        "lef_path": root / "design.lef",
        "def_path": root / "design.def",
        "sdc_path": root / "design.sdc",
    }
```

- [ ] **Step 8: Run the benchmark catalog tests**

Run: `pytest tests/chip/test_ispd_benchmarks.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/ispd_benchmarks.py tests/chip/test_ispd_benchmarks.py
git commit -m "feat: add ispd benchmark catalog"
```

### Task 2: Add Tcl Placement Injection Support

**Files:**
- Modify: `chip/openroad_bridge.py`
- Modify: `tests/chip/test_openroad_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.openroad_bridge import OpenROADBridge


def test_build_script_injects_place_inst_commands_for_movable_instances():
    bridge = OpenROADBridge()
    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"U1": (10.0, 20.0), "U2": (30.0, 40.0)},
        placement_mode="sa_injected",
        orientations={"U1": "N", "U2": "FS"},
    )

    assert "place_inst U1 10.0 20.0 N" in script
    assert "place_inst U2 30.0 40.0 FS" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_openroad_bridge.py::test_build_script_injects_place_inst_commands_for_movable_instances -v`
Expected: FAIL because `build_script` does not accept placement mode or emit `place_inst`

- [ ] **Step 3: Write minimal implementation**

```python
def build_script(..., placement_mode: str = "sa_injected", orientations: dict[str, str] | None = None) -> str:
    ...
    if placement_mode == "sa_injected":
        for inst_name, (x, y) in placement.items():
            orient = (orientations or {}).get(inst_name, "N")
            lines.append(f"place_inst {inst_name} {x} {y} {orient}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_openroad_bridge.py::test_build_script_injects_place_inst_commands_for_movable_instances -v`
Expected: PASS

- [ ] **Step 5: Add a fairness test for the shared tail flow**

```python
from chip.openroad_bridge import OpenROADBridge


def test_build_script_uses_same_tail_flow_for_built_in_and_sa():
    bridge = OpenROADBridge()

    built_in = bridge.build_script("design.lef", "design.def", {}, placement_mode="built_in")
    sa = bridge.build_script("design.lef", "design.def", {"U1": (1.0, 2.0)}, placement_mode="sa_injected")

    assert "global_route" in built_in
    assert "global_route" in sa
    assert built_in.split("global_route", maxsplit=1)[1] == sa.split("global_route", maxsplit=1)[1]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_openroad_bridge.py::test_build_script_uses_same_tail_flow_for_built_in_and_sa -v`
Expected: FAIL until the built-in and injected flows share one tail section

- [ ] **Step 7: Refactor `build_script` to use one tail template**

```python
tail_lines = ["global_route", "report_tns", "report_wns"]
...
lines.extend(tail_lines)
```

- [ ] **Step 8: Run the OpenROAD bridge tests**

Run: `pytest tests/chip/test_openroad_bridge.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/openroad_bridge.py tests/chip/test_openroad_bridge.py
git commit -m "feat: add place-inst OpenROAD injection"
```

### Task 3: Add OpenROAD Acceptance Evaluator

**Files:**
- Create: `chip/openroad_eval.py`
- Create: `tests/chip/test_openroad_eval.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.openroad_eval import build_evaluation_request


def test_build_evaluation_request_preserves_mode_and_artifacts():
    request = build_evaluation_request(
        benchmark_name="ibm01",
        lef_path="design.lef",
        def_path="design.def",
        placement_mode="built_in",
    )

    assert request.benchmark_name == "ibm01"
    assert request.placement_mode == "built_in"
    assert request.lef_path == "design.lef"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_openroad_eval.py::test_build_evaluation_request_preserves_mode_and_artifacts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chip.openroad_eval'`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationRequest:
    benchmark_name: str
    lef_path: str
    def_path: str
    placement_mode: str


def build_evaluation_request(...):
    return EvaluationRequest(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_openroad_eval.py::test_build_evaluation_request_preserves_mode_and_artifacts -v`
Expected: PASS

- [ ] **Step 5: Add a metrics-parsing test**

```python
from chip.openroad_eval import parse_openroad_metrics


def test_parse_openroad_metrics_extracts_routed_wirelength_timing_and_drc():
    metrics = parse_openroad_metrics(
        '''
        total routed wire length = 12345
        report_tns -12.5
        report_wns -0.9
        drc violation count = 7
        '''
    )

    assert metrics["total_routed_wirelength"] == 12345.0
    assert metrics["tns"] == -12.5
    assert metrics["wns"] == -0.9
    assert metrics["drc_violations"] == 7
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_openroad_eval.py::test_parse_openroad_metrics_extracts_routed_wirelength_timing_and_drc -v`
Expected: FAIL until metric parsing exists

- [ ] **Step 7: Implement the minimal parser**

```python
import re


def parse_openroad_metrics(stdout: str) -> dict[str, float | int]:
    ...
```

- [ ] **Step 8: Run the evaluator tests**

Run: `pytest tests/chip/test_openroad_eval.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/openroad_eval.py tests/chip/test_openroad_eval.py
git commit -m "feat: add OpenROAD acceptance evaluator"
```

### Task 4: Add Acceptance Runner

**Files:**
- Create: `chip/experiments/phase1_acceptance.py`
- Create: `tests/chip/test_phase1_acceptance.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase1_acceptance import summarize_acceptance_row


def test_summarize_acceptance_row_combines_baseline_and_openroad_metrics():
    row = summarize_acceptance_row(
        benchmark_name="ibm01",
        scale_bucket="small",
        design_stats={"node_count": 1000, "net_count": 1200},
        sa_summary={"hpwl": 10.0, "is_valid": True, "runtime_seconds": 1.5},
        built_in_metrics={"total_routed_wirelength": 100.0, "wns": -0.1, "tns": -1.0, "drc_violations": 0, "runtime_seconds": 5.0},
        sa_metrics={"total_routed_wirelength": 110.0, "wns": -0.2, "tns": -2.0, "drc_violations": 1, "runtime_seconds": 2.5},
    )

    assert row["benchmark"] == "ibm01"
    assert row["scale_bucket"] == "small"
    assert row["sa_hpwl"] == 10.0
    assert row["built_in_total_routed_wirelength"] == 100.0
    assert row["sa_drc_violations"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase1_acceptance.py::test_summarize_acceptance_row_combines_baseline_and_openroad_metrics -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chip.experiments.phase1_acceptance'`

- [ ] **Step 3: Write minimal implementation**

```python
def summarize_acceptance_row(...):
    return {
        ...
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase1_acceptance.py::test_summarize_acceptance_row_combines_baseline_and_openroad_metrics -v`
Expected: PASS

- [ ] **Step 5: Add a failing suite-orchestration test**

```python
from chip.experiments import phase1_acceptance as module


def test_run_acceptance_suite_surfaces_benchmark_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(
        module,
        "run_single_benchmark_acceptance",
        lambda *args, **kwargs: {"benchmark": "ibm01", "error": "routing_failed"},
    )

    rows = module.run_acceptance_suite(benchmark_root=tmp_path, output_path=tmp_path / "results.json")

    assert rows == [{"benchmark": "ibm01", "error": "routing_failed"}]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_phase1_acceptance.py::test_run_acceptance_suite_surfaces_benchmark_failures -v`
Expected: FAIL until the acceptance suite entrypoint exists

- [ ] **Step 7: Implement the minimal suite entrypoint**

```python
def run_acceptance_suite(...):
    ...
```

- [ ] **Step 8: Run the acceptance-runner tests**

Run: `pytest tests/chip/test_phase1_acceptance.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/experiments/phase1_acceptance.py tests/chip/test_phase1_acceptance.py
git commit -m "feat: add phase1 acceptance runner"
```

### Task 5: Wire Acceptance Flow To Current Baseline

**Files:**
- Modify: `chip/experiments/baseline.py`
- Modify: `chip/openroad_eval.py`
- Modify: `chip/experiments/phase1_acceptance.py`
- Modify: `tests/chip/test_baseline_experiment.py`
- Modify: `tests/chip/test_openroad_eval.py`
- Modify: `tests/chip/test_phase1_acceptance.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from chip.experiments.baseline import run_baseline_experiment


def test_baseline_experiment_returns_sa_runtime_seconds():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"

    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        bridge_binary="definitely-missing-openroad",
        iterations=1,
    )

    assert "runtime_seconds" in result["placements"]["sa"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_baseline_experiment.py::test_baseline_experiment_returns_sa_runtime_seconds -v`
Expected: FAIL until the SA summary includes runtime

- [ ] **Step 3: Implement the minimal runtime capture**

```python
import time

start = time.perf_counter()
sa_result = ...
sa_runtime_seconds = time.perf_counter() - start
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_baseline_experiment.py::test_baseline_experiment_returns_sa_runtime_seconds -v`
Expected: PASS

- [ ] **Step 5: Add a failing evaluator fairness test**

```python
from chip.openroad_eval import build_openroad_script


def test_build_openroad_script_changes_only_the_injection_section():
    built_in = build_openroad_script("design.lef", "design.def", placement_mode="built_in")
    injected = build_openroad_script(
        "design.lef",
        "design.def",
        placement_mode="sa_injected",
        placement={"U1": (1.0, 2.0)},
        orientations={"U1": "N"},
    )

    assert "place_inst U1 1.0 2.0 N" in injected
    assert built_in.split("global_route", maxsplit=1)[1] == injected.split("global_route", maxsplit=1)[1]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_openroad_eval.py::test_build_openroad_script_changes_only_the_injection_section -v`
Expected: FAIL until `chip.openroad_eval` exposes the shared-flow script builder

- [ ] **Step 7: Implement the shared script builder and route acceptance through it**

```python
def build_openroad_script(...):
    ...
```

- [ ] **Step 8: Run the affected test files**

Run: `pytest tests/chip/test_baseline_experiment.py tests/chip/test_openroad_eval.py tests/chip/test_phase1_acceptance.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/experiments/baseline.py chip/openroad_eval.py chip/experiments/phase1_acceptance.py tests/chip/test_baseline_experiment.py tests/chip/test_openroad_eval.py tests/chip/test_phase1_acceptance.py
git commit -m "feat: wire phase1 acceptance flow"
```

### Task 6: Run Acceptance Verification

**Files:**
- Verify: `tests/`
- Verify: `experiments/chip/`

- [ ] **Step 1: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 2: Run the Phase 1 acceptance command**

Run: `python -m chip.experiments.phase1_acceptance`
Expected: one results artifact written under `experiments/chip/`

- [ ] **Step 3: Inspect the artifact**

Run: `@'
import json
from pathlib import Path

path = Path("experiments/chip/phase1_acceptance_results.json")
rows = json.loads(path.read_text())
assert len(rows) == 3
for row in rows:
    assert "built_in_total_routed_wirelength" in row or "error" in row
    assert "sa_total_routed_wirelength" in row or "error" in row
print("phase1 acceptance artifact verified")
'@ | python -`
Expected: prints `phase1 acceptance artifact verified`

- [ ] **Step 4: Commit the acceptance results and code**

```bash
git add chip tests experiments docs
git commit -m "feat: complete phase1 roadmap acceptance"
```
