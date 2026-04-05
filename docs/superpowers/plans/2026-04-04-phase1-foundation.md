# Phase 1 Foundation Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the repository and implement the neutral engine foundation needed to validate Phase 1 with `ProblemGraph`, protocol contracts, and two toy domains.

**Architecture:** Build the smallest reusable core first: a domain-neutral `engine` package with protocol-based contracts and a lightweight graph container, then prove the extension point using an offline spatial toy and an online scheduling toy. Keep all domain semantics inside toy modules so the engine stays clean.

**Tech Stack:** Python 3.11, pytest, typing.Protocol, dataclasses

---

## File Map

- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `engine/__init__.py`
- Create: `engine/contracts.py`
- Create: `engine/graph.py`
- Create: `toys/__init__.py`
- Create: `toys/spatial/__init__.py`
- Create: `toys/spatial/problem.py`
- Create: `toys/online_scheduler/__init__.py`
- Create: `toys/online_scheduler/problem.py`
- Create: `tests/engine/test_problem_graph.py`
- Create: `tests/engine/test_contracts.py`
- Create: `tests/toys/test_spatial_toy.py`
- Create: `tests/toys/test_online_scheduler_toy.py`

### Task 1: Initialize Repository And Python Project

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `engine/__init__.py`
- Create: `toys/__init__.py`

- [ ] **Step 1: Initialize git repository and branch**

Run:

```bash
git init -b main
git checkout -b codex/phase1-foundation
```

Expected: repository initialized and current branch is `codex/phase1-foundation`

- [ ] **Step 2: Create the bootstrap files**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "graph-placement-engine"
version = "0.1.0"
description = "Domain-agnostic graph optimization engine foundations"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

```gitignore
# .gitignore
__pycache__/
.pytest_cache/
.worktrees/
worktrees/
*.pyc
```

```markdown
# README.md

This repository contains the Phase 1 foundations for a domain-agnostic graph optimization engine.
```

```python
# engine/__init__.py
"""Neutral engine package."""
```

```python
# toys/__init__.py
"""Toy domains used to validate engine neutrality."""
```

- [ ] **Step 3: Verify pytest can discover an empty test suite**

Run:

```bash
pytest
```

Expected: exit code `5` with output indicating no tests were collected

- [ ] **Step 4: Commit the bootstrap**

```bash
git add .gitignore pyproject.toml README.md engine/__init__.py toys/__init__.py
git commit -m "chore: bootstrap repository"
```

### Task 2: Add `ProblemGraph` With TDD

**Files:**
- Create: `engine/graph.py`
- Test: `tests/engine/test_problem_graph.py`

- [ ] **Step 1: Write the failing test**

```python
from engine.graph import ProblemGraph


def test_problem_graph_preserves_nodes_edges_and_global_features():
    graph = ProblemGraph(
        nodes={"n1": {"kind": "macro", "width": 3}},
        edges=[("n1", "n1", {"weight": 1.0})],
        global_features={"domain": "chip"},
    )

    assert graph.node_ids == ("n1",)
    assert graph.edge_count == 1
    assert graph.global_features["domain"] == "chip"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/engine/test_problem_graph.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `engine.graph`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemGraph:
    nodes: dict[str, dict]
    edges: list[tuple[str, str, dict]] = field(default_factory=list)
    global_features: dict[str, object] = field(default_factory=dict)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self.nodes.keys())

    @property
    def edge_count(self) -> int:
        return len(self.edges)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/engine/test_problem_graph.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/graph.py tests/engine/test_problem_graph.py
git commit -m "feat: add problem graph container"
```

### Task 3: Add Protocol Contracts With TDD

**Files:**
- Create: `engine/contracts.py`
- Test: `tests/engine/test_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import dataclass

from engine.contracts import ConstraintChecker, Evaluator, SurrogateModel, TransitionModel


@dataclass
class DemoTransition:
    def transition(self, state, action):
        return {"state": state, "action": action}


@dataclass
class DemoConstraintChecker:
    def check(self, state):
        return {"valid": True, "violations": []}


@dataclass
class DemoEvaluator:
    def evaluate(self, candidate):
        return {"score": 1.0}


@dataclass
class DemoSurrogate:
    def predict(self, observation):
        return {"score": 0.5}


def test_demo_objects_satisfy_engine_protocols():
    transition: TransitionModel = DemoTransition()
    checker: ConstraintChecker = DemoConstraintChecker()
    evaluator: Evaluator = DemoEvaluator()
    surrogate: SurrogateModel = DemoSurrogate()

    assert transition.transition({}, {})["action"] == {}
    assert checker.check({})["valid"] is True
    assert evaluator.evaluate({})["score"] == 1.0
    assert surrogate.predict({})["score"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/engine/test_contracts.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `engine.contracts`

- [ ] **Step 3: Write minimal implementation**

```python
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TransitionModel(Protocol):
    def transition(self, state: Any, action: Any) -> Any: ...


@runtime_checkable
class ConstraintChecker(Protocol):
    def check(self, state: Any) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, candidate: Any) -> Any: ...


@runtime_checkable
class SurrogateModel(Protocol):
    def predict(self, observation: Any) -> Any: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/engine/test_contracts.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/contracts.py tests/engine/test_contracts.py
git commit -m "feat: add engine protocol contracts"
```

### Task 4: Validate Offline Spatial Toy With TDD

**Files:**
- Create: `toys/spatial/__init__.py`
- Create: `toys/spatial/problem.py`
- Test: `tests/toys/test_spatial_toy.py`

- [ ] **Step 1: Write the failing test**

```python
from toys.spatial.problem import build_spatial_problem


def test_spatial_toy_builds_problem_graph_without_engine_changes():
    graph = build_spatial_problem(
        rectangles={
            "a": {"width": 1, "height": 1, "x": 0, "y": 0},
            "b": {"width": 1, "height": 1, "x": 2, "y": 0},
        }
    )

    assert graph.node_ids == ("a", "b")
    assert graph.global_features["objective"] == "pairwise_distance"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/toys/test_spatial_toy.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `toys.spatial.problem`

- [ ] **Step 3: Write minimal implementation**

```python
from engine.graph import ProblemGraph


def build_spatial_problem(rectangles: dict[str, dict]) -> ProblemGraph:
    return ProblemGraph(
        nodes=rectangles,
        edges=[],
        global_features={"objective": "pairwise_distance", "mode": "offline"},
    )
```

```python
# toys/spatial/__init__.py
"""Offline spatial toy domain."""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/toys/test_spatial_toy.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add toys/spatial/__init__.py toys/spatial/problem.py tests/toys/test_spatial_toy.py
git commit -m "feat: add spatial toy domain"
```

### Task 5: Validate Online Scheduler Toy With TDD

**Files:**
- Create: `toys/online_scheduler/__init__.py`
- Create: `toys/online_scheduler/problem.py`
- Test: `tests/toys/test_online_scheduler_toy.py`

- [ ] **Step 1: Write the failing test**

```python
from toys.online_scheduler.problem import build_scheduler_problem


def test_online_scheduler_toy_builds_problem_graph_without_engine_changes():
    graph = build_scheduler_problem(
        jobs={
            "j1": {"arrival": 0, "size": 3, "deadline": 4},
            "j2": {"arrival": 1, "size": 2, "deadline": 5},
        },
        machine_count=2,
    )

    assert graph.node_ids == ("j1", "j2")
    assert graph.global_features["mode"] == "online"
    assert graph.global_features["machine_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/toys/test_online_scheduler_toy.py -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `toys.online_scheduler.problem`

- [ ] **Step 3: Write minimal implementation**

```python
from engine.graph import ProblemGraph


def build_scheduler_problem(jobs: dict[str, dict], machine_count: int) -> ProblemGraph:
    return ProblemGraph(
        nodes=jobs,
        edges=[],
        global_features={
            "objective": "weighted_tardiness",
            "mode": "online",
            "machine_count": machine_count,
        },
    )
```

```python
# toys/online_scheduler/__init__.py
"""Online scheduling toy domain."""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/toys/test_online_scheduler_toy.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add toys/online_scheduler/__init__.py toys/online_scheduler/problem.py tests/toys/test_online_scheduler_toy.py
git commit -m "feat: add online scheduler toy domain"
```

### Task 6: Run Foundation Verification

**Files:**
- Verify: `tests/engine/test_problem_graph.py`
- Verify: `tests/engine/test_contracts.py`
- Verify: `tests/toys/test_spatial_toy.py`
- Verify: `tests/toys/test_online_scheduler_toy.py`

- [ ] **Step 1: Run the full targeted suite**

Run:

```bash
pytest tests/engine tests/toys -v
```

Expected: all tests PASS

- [ ] **Step 2: Commit the verified foundation slice**

```bash
git add engine toys tests
git commit -m "feat: complete phase1 foundation slice"
```
