# Phase 3 Roadmap Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the highest-value remaining gaps between the current Phase 3 implementation and the roadmap Phase 3 acceptance shape.

**Architecture:** Keep the existing PPO/search/imitation stack intact, add a roadmap-facing reward path using the current production proxy interface, and add a final report harness that emits the Phase 3 results surface in one canonical artifact. Avoid new search or reward inventions in this slice.

**Tech Stack:** Python, PyTorch, existing PPO/search harness, existing chip adapter/env, pytest, JSON experiment artifacts

---

## File Structure

- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\rl_adapter.py`
  - add a roadmap-facing proxy-delta reward mode
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_rl_adapter.py`
  - verify proxy-delta reward behavior
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_final_report.py`
  - generate a canonical Phase 3 results artifact from the current ladder/comparison surface
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_final_report.py`
  - verify report structure and roadmap-facing fields

### Task 1: Roadmap Proxy Reward Path

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\rl_adapter.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_rl_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
def test_chip_rl_adapter_proxy_delta_uses_delta_not_absolute_cost():
    adapter = ChipRLAdapter(problem=build_problem(), proxy=StubProxy(prediction=10.0), grid_shape=(4, 4))
    state = adapter.initial_state()
    next_state = adapter.apply_action(state, {"macro": "u1", "target": (4.0, 0.0)})
    reward = adapter.compute_reward(state, next_state, "proxy_delta")
    assert isinstance(reward, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_rl_adapter.py -q`
Expected: FAIL with `Unknown reward source: proxy_delta`

- [ ] **Step 3: Write minimal implementation**

```python
if reward_source == "proxy_delta":
    return self.proxy_delta(previous_state, next_state, len(previous_state.placed_macros))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_rl_adapter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/rl_adapter.py tests/chip/test_rl_adapter.py
git commit -m "feat: add roadmap proxy delta reward"
```

### Task 2: Canonical Phase 3 Final Report

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_final_report.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_final_report.py`

- [ ] **Step 1: Write the failing test**

```python
from chip.experiments.phase3_final_report import build_phase3_final_report


def test_build_phase3_final_report_contains_roadmap_sections():
    report = build_phase3_final_report()
    assert "roadmap_phase3_status" in report
    assert "benchmarks" in report
    assert "methods" in report
    assert "remaining_gaps" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chip/test_phase3_final_report.py -q`
Expected: FAIL with import error for `chip.experiments.phase3_final_report`

- [ ] **Step 3: Write minimal implementation**

```python
def build_phase3_final_report():
    return {
        "roadmap_phase3_status": "partial",
        "benchmarks": ...,
        "methods": ...,
        "remaining_gaps": ...,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chip/test_phase3_final_report.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_final_report.py tests/chip/test_phase3_final_report.py
git commit -m "feat: add phase3 final report harness"
```

### Task 3: Final Verification

**Files:**
- Verify existing and new files only

- [ ] **Step 1: Run focused suite**

Run: `pytest tests/chip/test_rl_adapter.py tests/chip/test_phase3_final_report.py -q`
Expected: PASS

- [ ] **Step 2: Run full suite**

Run: `pytest tests/engine tests/chip -q`
Expected: PASS

- [ ] **Step 3: Commit if needed**

```bash
git add docs/superpowers/task-memory.md
git commit -m "docs: update phase3 roadmap closeout memory"
```

## Self-Review

Spec coverage:

- roadmap-facing reward path: Task 1
- canonical final report artifact: Task 2
- verification: Task 3

Placeholder scan:

- no placeholder APIs remain
- all tasks name concrete files and commands

Type consistency:

- reward source string is `proxy_delta`
- final report keys are `roadmap_phase3_status`, `benchmarks`, `methods`, `remaining_gaps`
