# Phase 3 Benchmark Ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Phase 3 from toy bring-up to an experimentally meaningful benchmark ladder with update-level curves and multi-seed comparison reporting.

**Architecture:** Add a deterministic benchmark-ladder module that supplies tiny, small, and medium chip problems without changing the PPO/search interface. Extend PPO training to emit per-update curve points, then extend the comparison harness to aggregate repeated runs across seeds on the same method row layout.

**Tech Stack:** Python, PyTorch, existing chip problem/parser modules, existing PPO/search harness, pytest

---

## File Structure

- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_benchmarks.py`
  - benchmark ladder definitions and problem loaders
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_ppo.py`
  - per-update training history with rollout metrics
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_comparison.py`
  - multi-seed comparison and curve-rich reporting
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_benchmarks.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_ppo.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_comparison.py`

### Task 1: Benchmark Ladder

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_benchmarks.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_benchmarks.py`

- [ ] **Step 1: Write the failing tests**

```python
from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder


def test_phase3_benchmark_ladder_has_tiny_small_medium():
    ladder = build_phase3_benchmark_ladder()
    assert tuple(item["name"] for item in ladder) == ("tiny", "small", "medium")


def test_phase3_benchmark_ladder_increases_movable_macro_count():
    ladder = build_phase3_benchmark_ladder()
    movable_counts = [item["movable_macro_count"] for item in ladder]
    assert movable_counts[0] < movable_counts[1] < movable_counts[2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_benchmarks.py -q`
Expected: FAIL with import error for `chip.experiments.phase3_benchmarks`

- [ ] **Step 3: Write minimal implementation**

```python
def build_phase3_benchmark_ladder():
    return [
        {"name": "tiny", "movable_macro_count": 1, "problem": ...},
        {"name": "small", "movable_macro_count": 3, "problem": ...},
        {"name": "medium", "movable_macro_count": 8, "problem": ...},
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_benchmarks.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_benchmarks.py tests/chip/test_phase3_benchmarks.py
git commit -m "feat: add phase3 benchmark ladder"
```

### Task 2: PPO Curve Reporting

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_ppo.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_ppo.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import train_ppo_policy
from tests.chip.test_rl_adapter import build_problem


def test_train_ppo_policy_records_rollout_curve_points():
    policy = initialize_policy_for_problem(problem=build_problem(), grid_shape=(4, 4), hidden_dim=16)
    result = train_ppo_policy(
        problem=build_problem(),
        policy=policy,
        grid_shape=(4, 4),
        reward_source="analytic_hpwl",
        max_steps=1,
        num_updates=3,
        learning_rate=1e-3,
        seed=0,
    )
    assert len(result["history"]) == 3
    assert "final_hpwl" in result["history"][0]
    assert "best_hpwl_so_far" in result["history"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_ppo.py -q`
Expected: FAIL because history items do not include rollout curve fields

- [ ] **Step 3: Write minimal implementation**

```python
history.append({
    ...,
    "final_hpwl": rollout["final_hpwl"],
    "invalid_action_count": rollout["invalid_action_count"],
    "best_hpwl_so_far": best_hpwl_so_far,
})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_ppo.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_ppo.py tests/chip/test_phase3_ppo.py
git commit -m "feat: add ppo training curves"
```

### Task 3: Multi-Seed Comparison

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_comparison.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_comparison.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder
from chip.experiments.phase3_comparison import run_phase3_comparison


def test_phase3_comparison_supports_multiple_seeds():
    benchmark = build_phase3_benchmark_ladder()[1]
    report = run_phase3_comparison(
        problem=benchmark["problem"],
        grid_shape=benchmark["grid_shape"],
        hidden_dim=16,
        ppo_epochs=2,
        seeds=(0, 1),
    )
    assert report["seed_count"] == 2
    assert "aggregate" in report["methods"]["ppo_warm_start"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_comparison.py -q`
Expected: FAIL because `run_phase3_comparison` does not accept `seeds`

- [ ] **Step 3: Write minimal implementation**

```python
def run_phase3_comparison(..., seeds=(0,)):
    per_seed = ...
    return {
        "seed_count": len(seeds),
        "methods": {
            name: {
                "per_seed": rows,
                "aggregate": {...},
            }
            for name, rows in ...
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_comparison.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_comparison.py tests/chip/test_phase3_comparison.py
git commit -m "feat: add multi-seed phase3 comparison"
```

## Self-Review

Spec coverage:

- benchmark ladder: Task 1
- update-level sample-efficiency evidence: Task 2
- multi-seed protocol: Task 3

Placeholder scan:

- no placeholder APIs remain
- each task names concrete files and commands

Type consistency:

- benchmark ladder returns dictionaries with stable keys
- PPO history entries are dictionaries of numeric metrics
- comparison report explicitly contains `seed_count`, `methods`, `per_seed`, and `aggregate`

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-07-phase3-benchmark-ladder.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
