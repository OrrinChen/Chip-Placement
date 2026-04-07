# Phase 3 Search And Imitation Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real search baseline, reusable imitation-learning trajectory format, and PPO warm-start plumbing without destabilizing the existing Phase 3 PPO bring-up path.

**Architecture:** Keep search, imitation, and PPO as separate layers. Search reuses the existing adapter and action-mask contract to produce strong baselines and demonstration trajectories. Behavior cloning trains the existing PPO policy architecture in supervised mode from serialized search trajectories. PPO then optionally loads the BC checkpoint as a warm-start while keeping its value head fresh.

**Tech Stack:** Python, PyTorch, existing `chip.rl_adapter`, existing `engine.ppo_policy`, pytest

---

## File Structure

- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_baselines.py`
  - extend simple baselines into richer search-facing utilities
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_search.py`
  - deterministic beam search baseline over the same action space as PPO
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\imitation.py`
  - trajectory schema, save/load helpers, dataset preparation
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_imitation.py`
  - supervised behavior-cloning loop and checkpoint export
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\engine\ppo_policy.py`
  - expose helper methods for supervised action logits and checkpoint loading if needed
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_ppo.py`
  - PPO runner stub that can optionally load BC initialization
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_baselines.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_search.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_imitation.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_imitation.py`
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_ppo.py`

### Task 1: Beam Search Baseline

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_search.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_baselines.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_search.py`

- [ ] **Step 1: Write the failing tests**

```python
from chip.experiments.phase3_search import run_beam_search_baseline
from tests.chip.test_rl_adapter import make_problem


def test_beam_search_avoids_invalid_actions():
    problem = make_problem()
    result = run_beam_search_baseline(problem=problem, grid_shape=(4, 4), beam_width=2)
    assert result["invalid_action_count"] == 0


def test_beam_search_matches_or_beats_greedy_hpwl():
    from chip.experiments.phase3_baselines import run_greedy_hpwl_baseline

    problem = make_problem()
    greedy = run_greedy_hpwl_baseline(problem=problem, grid_shape=(4, 4))
    beam = run_beam_search_baseline(problem=problem, grid_shape=(4, 4), beam_width=2)
    assert beam["final_hpwl"] <= greedy["final_hpwl"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chip/test_phase3_search.py -q`
Expected: FAIL with import error for `chip.experiments.phase3_search`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass

from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter


@dataclass(frozen=True)
class SearchCandidate:
    state: object
    trace: tuple[dict[str, object], ...]
    score: float


def run_beam_search_baseline(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    beam_width: int = 4,
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    initial_state = adapter.initial_state()
    frontier = [SearchCandidate(state=initial_state, trace=tuple(), score=adapter.current_hpwl(initial_state))]
    invalid_action_count = 0

    while True:
        next_frontier: list[SearchCandidate] = []
        expanded = False
        for candidate in frontier:
            step_index = len(candidate.state.placed_macros)
            current_macro = adapter.current_macro_for_step(step_index)
            if current_macro is None:
                next_frontier.append(candidate)
                continue
            action_mask = adapter.build_action_mask(candidate.state, step_index=step_index, grid_shape=grid_shape)
            for row_index in range(grid_shape[0]):
                for col_index in range(grid_shape[1]):
                    if bool(action_mask[row_index, col_index]):
                        continue
                    expanded = True
                    target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
                    action = {"macro": current_macro, "target": target}
                    next_state = adapter.apply_action(candidate.state, action)
                    next_trace = candidate.trace + ({"action": action, "step_index": step_index},)
                    next_score = adapter.current_hpwl(next_state)
                    next_frontier.append(SearchCandidate(state=next_state, trace=next_trace, score=next_score))
        if not expanded:
            break
        next_frontier.sort(key=lambda candidate: candidate.score)
        frontier = next_frontier[:beam_width]

    best = min(frontier, key=lambda candidate: candidate.score)
    return {
        "steps": len(best.trace),
        "trace": list(best.trace),
        "invalid_action_count": invalid_action_count,
        "final_hpwl": adapter.current_hpwl(best.state),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chip/test_phase3_search.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_search.py tests/chip/test_phase3_search.py
git commit -m "feat: add phase3 beam search baseline"
```

### Task 2: Demonstration Schema And IO

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\imitation.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_search.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_imitation.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from chip.imitation import DemonstrationStep, load_demonstrations, save_demonstrations


def test_demonstration_round_trip(tmp_path: Path):
    steps = [
        DemonstrationStep(
            design="toy",
            rollout_type="beam_search",
            step_index=0,
            current_macro="U1",
            action_index=3,
            action_mask=[False, False, True, False],
            final_score=12.5,
        )
    ]
    path = tmp_path / "demo.json"
    save_demonstrations(path, steps)
    loaded = load_demonstrations(path)
    assert loaded == steps
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chip/test_imitation.py -q`
Expected: FAIL with import error for `chip.imitation`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemonstrationStep:
    design: str
    rollout_type: str
    step_index: int
    current_macro: str
    action_index: int
    action_mask: list[bool]
    final_score: float


def save_demonstrations(path: Path, steps: list[DemonstrationStep]) -> None:
    path.write_text(json.dumps([asdict(step) for step in steps], indent=2), encoding="utf-8")


def load_demonstrations(path: Path) -> list[DemonstrationStep]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [DemonstrationStep(**item) for item in raw]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chip/test_imitation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/imitation.py tests/chip/test_imitation.py
git commit -m "feat: add imitation trajectory schema"
```

### Task 3: Behavior Cloning Trainer

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_imitation.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\engine\ppo_policy.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_imitation.py`

- [ ] **Step 1: Write the failing tests**

```python
import torch

from chip.experiments.phase3_imitation import masked_behavior_cloning_loss


def test_masked_behavior_cloning_ignores_invalid_actions():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    actions = torch.tensor([1], dtype=torch.int64)
    invalid_mask = torch.tensor([[False, False, True, True]])
    loss = masked_behavior_cloning_loss(logits, actions, invalid_mask)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chip/test_phase3_imitation.py -q`
Expected: FAIL with import error for `chip.experiments.phase3_imitation`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_behavior_cloning_loss(
    logits: torch.Tensor,
    action_index: torch.Tensor,
    invalid_action_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(invalid_action_mask, float("-inf"))
    return F.cross_entropy(masked_logits, action_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chip/test_phase3_imitation.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_imitation.py tests/chip/test_phase3_imitation.py
git commit -m "feat: add behavior cloning loss"
```

### Task 4: PPO Warm-Start Plumbing

**Files:**
- Create: `C:\Users\14574\Chip\.worktrees\phase3-foundation\chip\experiments\phase3_ppo.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase3-foundation\engine\ppo_policy.py`
- Test: `C:\Users\14574\Chip\.worktrees\phase3-foundation\tests\chip\test_phase3_ppo.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import torch

from chip.experiments.phase3_ppo import load_policy_warm_start
from engine.ppo_policy import PPOPlacementPolicy


def test_load_policy_warm_start_ignores_missing_value_head(tmp_path: Path):
    policy = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=0,
        grid_channels=6,
        grid_shape=(4, 4),
    )
    checkpoint = {"policy_state_dict": {key: value for key, value in policy.state_dict().items() if not key.startswith("value_head.")}}
    path = tmp_path / "bc.pt"
    torch.save(checkpoint, path)
    loaded = load_policy_warm_start(policy, path)
    assert loaded is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/chip/test_phase3_ppo.py -q`
Expected: FAIL with import error for `chip.experiments.phase3_ppo`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path

import torch


def load_policy_warm_start(policy, checkpoint_path: Path) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["policy_state_dict"]
    policy.load_state_dict(state_dict, strict=False)
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/chip/test_phase3_ppo.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_ppo.py tests/chip/test_phase3_ppo.py
git commit -m "feat: add ppo warm start loader"
```

## Self-Review

Spec coverage:

- search baseline: Task 1
- demonstration format: Task 2
- behavior cloning: Task 3
- PPO warm-start: Task 4

Placeholder scan:

- no `TODO`, `TBD`, or cross-task placeholder references remain

Type consistency:

- `DemonstrationStep`
- `masked_behavior_cloning_loss`
- `load_policy_warm_start`

These names are defined in the tasks that first introduce them and reused consistently later.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-07-phase3-search-imitation-hybrid.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
