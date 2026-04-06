# Phase 2 Foundation And Data Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize the neutral engine API, close the SA legality gap with a lightweight legalization pass, and launch the Phase 2 data-generation pipeline for GNN proxy training.

**Architecture:** Keep the engine surface neutral and protocol-first, push legality and OpenROAD interaction down into `chip/`, and make data generation a reproducible experiment pipeline that can run long OpenROAD jobs independently of model-training code.

**Tech Stack:** Python 3.11, pytest, dataclasses, typing.Protocol, pathlib, json, Dockerized OpenROAD, Tcl script generation

---

## File Map

- Modify: `engine/contracts.py`
- Modify: `engine/graph.py`
- Modify: `engine/__init__.py`
- Create: `tests/engine/test_contracts.py`
- Modify: `tests/engine/test_problem_graph.py`
- Modify: `chip/netlist_parser.py`
- Modify: `chip/openroad_bridge.py`
- Modify: `chip/experiments/baseline.py`
- Create: `chip/legalization.py`
- Create: `tests/chip/test_legalization.py`
- Create: `chip/experiments/phase2_dataset.py`
- Create: `tests/chip/test_phase2_dataset.py`
- Modify: `docs/superpowers/task-memory.md`

### Task 1: Stabilize Neutral Engine Naming And Contracts

**Files:**
- Modify: `engine/contracts.py`
- Modify: `engine/graph.py`
- Modify: `engine/__init__.py`
- Modify: `tests/engine/test_contracts.py`
- Modify: `tests/engine/test_problem_graph.py`

- [ ] **Step 1: Write the failing contract test**

```python
from dataclasses import dataclass

from engine.contracts import (
    ConstraintChecker,
    DecisionEnv,
    Evaluator,
    SearchStrategy,
    SurrogateModel,
    TrainingStrategy,
)


@dataclass
class DemoDecisionEnv:
    def transition(self, state, action):
        return {"state": state, "action": action}


@dataclass
class DemoSearch:
    def solve(self, problem, objective):
        return {"status": "ok"}


@dataclass
class DemoTraining:
    def fit(self, samples):
        return {"rows": len(samples)}


def test_demo_objects_satisfy_engine_protocols():
    env: DecisionEnv = DemoDecisionEnv()
    search: SearchStrategy = DemoSearch()
    trainer: TrainingStrategy = DemoTraining()

    assert env.transition({}, {})["action"] == {}
    assert search.solve({}, None)["status"] == "ok"
    assert trainer.fit([1, 2])["rows"] == 2
```

- [ ] **Step 2: Run the contract test to verify it fails**

Run: `pytest tests/engine/test_contracts.py -v`
Expected: FAIL because `DecisionEnv`, `SearchStrategy`, or `TrainingStrategy` are missing

- [ ] **Step 3: Implement the thin protocol surface**

```python
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DecisionEnv(Protocol):
    def transition(self, state: Any, action: Any) -> Any: ...


@runtime_checkable
class SearchStrategy(Protocol):
    def solve(self, problem: Any, objective: Any) -> Any: ...


@runtime_checkable
class TrainingStrategy(Protocol):
    def fit(self, samples: Any) -> Any: ...
```

- [ ] **Step 4: Re-export the stable engine API**

```python
from engine.graph import ProblemGraph
from engine.contracts import (
    ConstraintChecker,
    DecisionEnv,
    Evaluator,
    SearchStrategy,
    SurrogateModel,
    TrainingStrategy,
)
```

- [ ] **Step 5: Run engine tests**

Run: `pytest tests/engine -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add engine tests/engine
git commit -m "feat: stabilize neutral engine contracts"
```

### Task 2: Add A Lightweight SA Legalization Pass

**Files:**
- Create: `chip/legalization.py`
- Modify: `chip/openroad_bridge.py`
- Modify: `chip/experiments/baseline.py`
- Modify: `chip/experiments/phase1_acceptance.py`
- Create: `tests/chip/test_legalization.py`
- Modify: `tests/chip/test_phase1_acceptance.py`

- [ ] **Step 1: Write the failing legalization test**

```python
from pathlib import Path

from chip.legalization import legalize_with_openroad


def test_legalize_with_openroad_requests_detailed_placement(tmp_path):
    artifact_dir = tmp_path / "bench"
    artifact_dir.mkdir()
    script = legalize_with_openroad(
        artifact_dir=artifact_dir,
        lef_paths=("tech.lef", "cells.lef"),
        def_path="seed.def",
        placement={"U1": (10.2, 2.0)},
        orientations={"U1": "N"},
    )

    assert "place_inst -name U1" in script
    assert "detailed_placement" in script
    assert "write_def legalized.def" in script
```

- [ ] **Step 2: Run the legalization test to verify it fails**

Run: `pytest tests/chip/test_legalization.py::test_legalize_with_openroad_requests_detailed_placement -v`
Expected: FAIL with `ModuleNotFoundError` for `chip.legalization`

- [ ] **Step 3: Implement the minimal legalizer helper**

```python
def build_legalization_script(...):
    return bridge.build_script(
        lef_paths,
        def_path,
        placement,
        placement_mode="sa_injected",
        orientations=orientations,
        use_detailed_placement=True,
        write_def_path="legalized.def",
    )
```

- [ ] **Step 4: Thread legalization through the acceptance path**

```python
legalized_def_path = legalize_with_openroad(...)
_, legalized_problem = load_chip_problem(prepared["cells_lef_path"], legalized_def_path)
sa_summary["is_valid"] = build_constraint_registry(legalized_problem).check_all(legalized_problem.placement()).is_valid
```

- [ ] **Step 5: Run the focused chip tests**

Run: `pytest tests/chip/test_legalization.py tests/chip/test_phase1_acceptance.py tests/chip/test_openroad_bridge.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chip tests/chip
git commit -m "feat: add openroad legalization pass"
```

### Task 3: Build The Phase 2 Dataset Sampling Pipeline

**Files:**
- Create: `chip/experiments/phase2_dataset.py`
- Modify: `chip/ispd_benchmarks.py`
- Create: `tests/chip/test_phase2_dataset.py`

- [ ] **Step 1: Write the failing dataset manifest test**

```python
from chip.experiments.phase2_dataset import build_sampling_plan


def test_build_sampling_plan_expands_random_and_sa_samples():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        random_samples=2,
        sa_truncation_steps=(1, 3),
    )

    assert [item["source"] for item in plan] == [
        "random_perturbation",
        "random_perturbation",
        "sa_truncated",
        "sa_truncated",
    ]
```

- [ ] **Step 2: Run the dataset test to verify it fails**

Run: `pytest tests/chip/test_phase2_dataset.py::test_build_sampling_plan_expands_random_and_sa_samples -v`
Expected: FAIL because `phase2_dataset.py` does not exist

- [ ] **Step 3: Implement the sampling-plan builder**

```python
def build_sampling_plan(benchmark_name, random_samples, sa_truncation_steps):
    rows = []
    for index in range(random_samples):
        rows.append({"benchmark": benchmark_name, "source": "random_perturbation", "sample_id": f"rand_{index:04d}"})
    for steps in sa_truncation_steps:
        rows.append({"benchmark": benchmark_name, "source": "sa_truncated", "sample_id": f"sa_{steps:04d}", "sa_iterations": steps})
    return rows
```

- [ ] **Step 4: Add dataset artifact layout and row schema**

```python
{
    "benchmark": "mgc_pci_bridge32_1",
    "sample_id": "rand_0001",
    "source": "random_perturbation",
    "placement_def": "samples/mgc_pci_bridge32_1/rand_0001.def",
    "metrics_json": "samples/mgc_pci_bridge32_1/rand_0001.metrics.json",
    "phase1_gate_status": "infra_pass / quality_pending",
}
```

- [ ] **Step 5: Run the dataset tests**

Run: `pytest tests/chip/test_phase2_dataset.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chip/experiments/phase2_dataset.py tests/chip/test_phase2_dataset.py chip/ispd_benchmarks.py
git commit -m "feat: scaffold phase2 dataset sampling"
```

### Task 4: Add Long-Running Ground-Truth Collection

**Files:**
- Modify: `chip/experiments/phase2_dataset.py`
- Modify: `chip/openroad_eval.py`
- Modify: `docs/superpowers/task-memory.md`
- Modify: `tests/chip/test_phase2_dataset.py`

- [ ] **Step 1: Write the failing collection test**

```python
from chip.experiments.phase2_dataset import summarize_dataset_row


def test_summarize_dataset_row_keeps_metrics_and_gate_state():
    row = summarize_dataset_row(
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation"},
        metrics={"total_routed_wirelength": 100.0, "wns": -0.1, "tns": -1.0, "drc_violations": 0, "runtime_seconds": 5.0},
        gate_status="infra_pass / quality_pending",
    )

    assert row["sample_id"] == "rand_0001"
    assert row["total_routed_wirelength"] == 100.0
    assert row["phase1_gate_status"] == "infra_pass / quality_pending"
```

- [ ] **Step 2: Run the collection test to verify it fails**

Run: `pytest tests/chip/test_phase2_dataset.py::test_summarize_dataset_row_keeps_metrics_and_gate_state -v`
Expected: FAIL until the summarizer exists

- [ ] **Step 3: Implement the minimal collector entrypoint**

```python
def collect_ground_truth(...):
    for sample in sampling_plan:
        metrics = evaluate_openroad_request(...)
        rows.append(summarize_dataset_row(sample=sample, metrics=metrics, gate_status=gate_status))
```

- [ ] **Step 4: Add one documented command for long runs**

Run target: `python -m chip.experiments.phase2_dataset --benchmark-root tmp --output experiments/chip/phase2_dataset_manifest.json`

- [ ] **Step 5: Run the dataset tests**

Run: `pytest tests/chip/test_phase2_dataset.py tests/chip/test_openroad_eval.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chip/experiments/phase2_dataset.py chip/openroad_eval.py tests/chip/test_phase2_dataset.py docs/superpowers/task-memory.md
git commit -m "feat: add phase2 ground truth collection"
```

### Task 5: Launch And Record The First Dataset Batch

**Files:**
- Verify: `experiments/chip/`
- Modify: `docs/superpowers/task-memory.md`

- [ ] **Step 1: Run the full chip test suite**

Run: `pytest tests/chip -v`
Expected: PASS

- [ ] **Step 2: Launch the first dataset batch**

Run: `python -m chip.experiments.phase2_dataset --benchmark-root tmp --output experiments/chip/phase2_dataset_manifest.json`
Expected: a manifest file plus per-sample metrics under `experiments/chip/`

- [ ] **Step 3: Verify the manifest structure**

Run:

```bash
@'
import json
from pathlib import Path

path = Path("experiments/chip/phase2_dataset_manifest.json")
rows = json.loads(path.read_text())
assert rows, "dataset manifest should not be empty"
assert "sample_id" in rows[0]
assert "phase1_gate_status" in rows[0]
print("phase2 dataset manifest verified")
'@ | python -
```

Expected: prints `phase2 dataset manifest verified`

- [ ] **Step 4: Update task memory with batch status**

```markdown
- Phase 2 dataset generation has started.
- The first manifest lives at `experiments/chip/phase2_dataset_manifest.json`.
- SA legality repair is either complete or still marked pending in the batch metadata.
```

- [ ] **Step 5: Commit**

```bash
git add experiments/chip docs/superpowers/task-memory.md
git commit -m "feat: launch first phase2 dataset batch"
```
