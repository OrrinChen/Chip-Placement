# Phase 1 Real Benchmark Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate realistic ORFS benchmark export, add a safe large-design baseline path, and generate a reproducible three-design Phase 1 summary.

**Architecture:** Keep all new work inside the chip domain and experiment layer. Add a focused exporter for ORFS placement artifacts, extend the baseline harness with explicit analytical skip metadata, and add a benchmark reporter that composes those pieces without changing `engine/` contracts.

**Tech Stack:** Python 3.11, pytest, pathlib, dataclasses, subprocess, Dockerized OpenROAD

---

## File Map

- Create: `chip/orfs_export.py`
- Create: `chip/experiments/real_benchmarks.py`
- Modify: `chip/experiments/baseline.py`
- Modify: `tests/chip/test_openroad_bridge.py`
- Modify: `tests/chip/test_baseline_experiment.py`
- Create: `tests/chip/test_orfs_export.py`
- Create: `tests/chip/test_real_benchmarks.py`

### Task 1: Add ORFS Export Utility

**Files:**
- Create: `chip/orfs_export.py`
- Test: `tests/chip/test_orfs_export.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from chip.orfs_export import ORFSDesignExporter


def test_exporter_builds_expected_output_paths(tmp_path):
    exporter = ORFSDesignExporter(work_root=tmp_path / "orfs-work", export_root=tmp_path / "orfs-export")

    bundle = exporter.bundle_paths("gcd")

    assert bundle.design_name == "gcd"
    assert bundle.odb_path == tmp_path / "orfs-work" / "results" / "nangate45" / "gcd" / "base" / "3_place.odb"
    assert bundle.output_dir == tmp_path / "orfs-export" / "nangate45" / "gcd"
    assert bundle.def_path == bundle.output_dir / "3_place.def"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_orfs_export.py::test_exporter_builds_expected_output_paths -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chip.orfs_export'`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ORFSExportBundle:
    design_name: str
    odb_path: Path
    output_dir: Path
    def_path: Path
    sdc_path: Path
    tech_lef_path: Path
    macro_lef_path: Path


class ORFSDesignExporter:
    def __init__(self, work_root: Path, export_root: Path, platform: str = "nangate45") -> None:
        self.work_root = Path(work_root)
        self.export_root = Path(export_root)
        self.platform = platform

    def bundle_paths(self, design_name: str) -> ORFSExportBundle:
        output_dir = self.export_root / self.platform / design_name
        return ORFSExportBundle(
            design_name=design_name,
            odb_path=self.work_root / "results" / self.platform / design_name / "base" / "3_place.odb",
            output_dir=output_dir,
            def_path=output_dir / "3_place.def",
            sdc_path=output_dir / "3_place.sdc",
            tech_lef_path=output_dir / "NangateOpenCellLibrary.tech.lef",
            macro_lef_path=output_dir / "NangateOpenCellLibrary.macro.mod.lef",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_orfs_export.py::test_exporter_builds_expected_output_paths -v`
Expected: PASS

- [ ] **Step 5: Add a command-construction test**

```python
def test_exporter_builds_docker_openroad_command(tmp_path):
    exporter = ORFSDesignExporter(
        work_root=tmp_path / "orfs-work",
        export_root=tmp_path / "orfs-export",
        docker_image="openroad/orfs",
    )

    bundle = exporter.bundle_paths("gcd")
    command = exporter.build_export_command(bundle, Path("/workspace/export.tcl"))

    assert command[0] == "docker"
    assert "openroad/orfs" in command
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_orfs_export.py::test_exporter_builds_docker_openroad_command -v`
Expected: FAIL with `AttributeError` for missing `build_export_command`

- [ ] **Step 7: Implement the minimal command builder**

```python
    def build_export_command(self, bundle: ORFSExportBundle, script_path: Path) -> list[str]:
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.export_root.parent.as_posix()}:/workspace",
            self.docker_image,
            self.docker_openroad_path,
            "-exit",
            script_path.as_posix(),
        ]
```

- [ ] **Step 8: Run exporter tests**

Run: `pytest tests/chip/test_orfs_export.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/orfs_export.py tests/chip/test_orfs_export.py
git commit -m "feat: add orfs export utility"
```

### Task 2: Add Large-Design Analytical Skip Metadata

**Files:**
- Modify: `chip/experiments/baseline.py`
- Modify: `tests/chip/test_baseline_experiment.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from chip.experiments.baseline import run_baseline_experiment


def test_baseline_experiment_skips_analytical_when_graph_is_too_large():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"

    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        analytical_node_limit=0,
        bridge_binary="definitely-missing-openroad",
    )

    assert result["placements"]["analytical"]["status"] == "skipped"
    assert result["placements"]["analytical"]["reason"] == "node_limit_exceeded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_baseline_experiment.py::test_baseline_experiment_skips_analytical_when_graph_is_too_large -v`
Expected: FAIL because the analytical placement summary does not expose skip metadata

- [ ] **Step 3: Write minimal implementation**

```python
def run_baseline_experiment(..., analytical_node_limit: int | None = 200) -> dict[str, object]:
    ...
    if analytical_node_limit is not None and len(graph.node_ids) > analytical_node_limit:
        analytical_summary = {
            "status": "skipped",
            "reason": "node_limit_exceeded",
            "node_count": len(graph.node_ids),
        }
    else:
        analytical = QuadraticPlacementSolver().solve(graph)
        analytical_summary = {
            "status": "completed",
            **_summarize(problem, constraints, analytical),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_baseline_experiment.py::test_baseline_experiment_skips_analytical_when_graph_is_too_large -v`
Expected: PASS

- [ ] **Step 5: Add a regression test for the completed path**

```python
def test_baseline_experiment_marks_analytical_completed_for_small_graph():
    benchmark_dir = Path(__file__).resolve().parents[2] / "chip" / "benchmarks"

    result = run_baseline_experiment(
        benchmark_dir / "sample.lef",
        benchmark_dir / "sample.def",
        analytical_node_limit=10,
        bridge_binary="definitely-missing-openroad",
    )

    assert result["placements"]["analytical"]["status"] == "completed"
    assert "hpwl" in result["placements"]["analytical"]
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_baseline_experiment.py::test_baseline_experiment_marks_analytical_completed_for_small_graph -v`
Expected: FAIL until the completed analytical summary includes status metadata

- [ ] **Step 7: Extend the implementation just enough to pass both tests**

```python
def _placement_summary(..., placement: dict[str, tuple[float, float]], status: str = "completed") -> dict[str, float | bool | str]:
    report = constraints.check_all(placement)
    return {
        "status": status,
        "hpwl": hpwl_cost(problem, placement),
        "total_penalty": report.total_penalty,
        "is_valid": report.is_valid,
    }
```

- [ ] **Step 8: Run the baseline test file**

Run: `pytest tests/chip/test_baseline_experiment.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/experiments/baseline.py tests/chip/test_baseline_experiment.py
git commit -m "feat: add large-design baseline fallback"
```

### Task 3: Add Real Benchmark Reporter

**Files:**
- Create: `chip/experiments/real_benchmarks.py`
- Create: `tests/chip/test_real_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.real_benchmarks import summarize_design_result


def test_summarize_design_result_extracts_phase1_metrics():
    result = {
        "placements": {
            "initial": {"hpwl": 10.0, "is_valid": True},
            "analytical": {"status": "skipped", "reason": "node_limit_exceeded"},
            "sa": {"hpwl": 8.0, "is_valid": False},
        },
        "openroad_available": True,
    }
    design_stats = {"node_count": 100, "fixed_node_count": 12, "net_count": 50}

    summary = summarize_design_result("gcd", design_stats, result)

    assert summary["design"] == "gcd"
    assert summary["initial_hpwl"] == 10.0
    assert summary["sa_hpwl"] == 8.0
    assert summary["analytical_status"] == "skipped"
    assert summary["openroad_available"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_real_benchmarks.py::test_summarize_design_result_extracts_phase1_metrics -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chip.experiments.real_benchmarks'`

- [ ] **Step 3: Write minimal implementation**

```python
def summarize_design_result(design_name: str, design_stats: dict[str, int], result: dict[str, object]) -> dict[str, object]:
    return {
        "design": design_name,
        "node_count": design_stats["node_count"],
        "fixed_node_count": design_stats["fixed_node_count"],
        "net_count": design_stats["net_count"],
        "initial_hpwl": result["placements"]["initial"]["hpwl"],
        "initial_is_valid": result["placements"]["initial"]["is_valid"],
        "sa_hpwl": result["placements"]["sa"]["hpwl"],
        "sa_is_valid": result["placements"]["sa"]["is_valid"],
        "analytical_status": result["placements"]["analytical"]["status"],
        "openroad_available": result["openroad_available"],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_real_benchmarks.py::test_summarize_design_result_extracts_phase1_metrics -v`
Expected: PASS

- [ ] **Step 5: Add a mixed-availability reporter test**

```python
from chip.experiments.real_benchmarks import summarize_many_designs


def test_summarize_many_designs_keeps_missing_design_errors_visible():
    rows = summarize_many_designs(
        {
            "gcd": {"summary": {"design": "gcd"}, "error": None},
            "aes": {"summary": None, "error": "missing_odb"},
        }
    )

    assert rows[0]["design"] == "gcd"
    assert rows[1]["design"] == "aes"
    assert rows[1]["error"] == "missing_odb"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_real_benchmarks.py::test_summarize_many_designs_keeps_missing_design_errors_visible -v`
Expected: FAIL until multi-design summarization exists

- [ ] **Step 7: Implement the minimal reporter helpers**

```python
def summarize_many_designs(results: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for design_name, payload in results.items():
        if payload["summary"] is not None:
            rows.append(payload["summary"])
        else:
            rows.append({"design": design_name, "error": payload["error"]})
    return rows
```

- [ ] **Step 8: Run the reporter test file**

Run: `pytest tests/chip/test_real_benchmarks.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/experiments/real_benchmarks.py tests/chip/test_real_benchmarks.py
git commit -m "feat: add real benchmark reporter"
```

### Task 4: Wire Reporter To Real Exported Designs

**Files:**
- Modify: `chip/orfs_export.py`
- Modify: `chip/experiments/real_benchmarks.py`
- Modify: `tests/chip/test_orfs_export.py`
- Modify: `tests/chip/test_real_benchmarks.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from chip.experiments.real_benchmarks import design_artifact_paths


def test_design_artifact_paths_points_to_exported_bundle(tmp_path):
    export_root = tmp_path / "orfs-export"

    paths = design_artifact_paths(export_root, "gcd")

    assert paths["lef_path"] == export_root / "nangate45" / "gcd" / "NangateOpenCellLibrary.tech.lef"
    assert paths["def_path"] == export_root / "nangate45" / "gcd" / "3_place.def"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_real_benchmarks.py::test_design_artifact_paths_points_to_exported_bundle -v`
Expected: FAIL until the reporter exposes exported artifact lookup

- [ ] **Step 3: Implement the minimal artifact lookup**

```python
def design_artifact_paths(export_root: Path, design_name: str, platform: str = "nangate45") -> dict[str, Path]:
    bundle_root = Path(export_root) / platform / design_name
    return {
        "lef_path": bundle_root / "NangateOpenCellLibrary.tech.lef",
        "def_path": bundle_root / "3_place.def",
        "macro_lef_path": bundle_root / "NangateOpenCellLibrary.macro.mod.lef",
        "sdc_path": bundle_root / "3_place.sdc",
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_real_benchmarks.py::test_design_artifact_paths_points_to_exported_bundle -v`
Expected: PASS

- [ ] **Step 5: Add an export-needed test**

```python
def test_exporter_reports_missing_odb_before_running(tmp_path):
    exporter = ORFSDesignExporter(work_root=tmp_path / "orfs-work", export_root=tmp_path / "orfs-export")

    bundle = exporter.bundle_paths("ibex")

    assert exporter.export_design("ibex")["status"] == "missing_odb"
    assert exporter.export_design("ibex")["bundle"] == bundle
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/chip/test_orfs_export.py::test_exporter_reports_missing_odb_before_running -v`
Expected: FAIL until the exporter exposes a non-throwing export status API

- [ ] **Step 7: Implement the minimal status-returning export method**

```python
    def export_design(self, design_name: str) -> dict[str, object]:
        bundle = self.bundle_paths(design_name)
        if not bundle.odb_path.exists():
            return {"status": "missing_odb", "bundle": bundle}
        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        return {"status": "ready", "bundle": bundle}
```

- [ ] **Step 8: Run the exporter and reporter tests**

Run: `pytest tests/chip/test_orfs_export.py tests/chip/test_real_benchmarks.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add chip/orfs_export.py chip/experiments/real_benchmarks.py tests/chip/test_orfs_export.py tests/chip/test_real_benchmarks.py
git commit -m "feat: wire real benchmark export paths"
```

### Task 5: Run Closeout Verification

**Files:**
- Verify: `tests/chip`
- Verify: `orfs-export/`
- Verify: `orfs-work/results/nangate45`

- [ ] **Step 1: Run the full suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 2: Export the three real designs**

Run: `python -m chip.experiments.real_benchmarks --design gcd --design aes --design ibex`
Expected: exporter creates or verifies bundles under `orfs-export/nangate45/`

- [ ] **Step 3: Generate the summary table**

Run: `python -m chip.experiments.real_benchmarks --summary-only`
Expected: JSON or tabular output containing `gcd`, `aes`, and `ibex`

- [ ] **Step 4: Commit the completed closeout work**

```bash
git add chip tests docs
git commit -m "feat: close out phase1 real benchmarks"
```
