# Phase 3 PPO Bring-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first executable Phase 3 PPO bring-up slice: PPO model skeleton, PPO-facing adapter observations, and masked random / greedy baselines.

**Architecture:** Keep the engine-level PPO model in `engine/ppo_policy.py`, keep chip-specific grid, seed-completion, and masking logic in `chip/rl_adapter.py`, and add small experiment helpers in `chip/experiments/` for masked-random and greedy baselines. This slice does not include PPO training yet; it prepares the exact state/action/reward path that training will consume.

**Tech Stack:** Python 3.11, PyTorch, existing `engine` and `chip` packages, pytest

---

## File Map

- Create: `engine/ppo_policy.py`
- Modify: `engine/__init__.py`
- Modify: `chip/rl_adapter.py`
- Create: `chip/experiments/phase3_baselines.py`
- Create: `tests/engine/test_ppo_policy.py`
- Modify: `tests/chip/test_rl_adapter.py`
- Create: `tests/chip/test_phase3_baselines.py`
- Modify: `docs/superpowers/task-memory.md`

### Task 1: Extend The RL Adapter For PPO Observations

**Files:**
- Modify: `chip/rl_adapter.py`
- Modify: `tests/chip/test_rl_adapter.py`

- [ ] **Step 1: Write the failing PPO observation tests**

```python
def test_chip_rl_adapter_observe_includes_current_macro_canvas_and_action_mask():
    adapter = ChipRLAdapter(problem=build_problem(), grid_shape=(4, 4))
    observation = adapter.observe(adapter.initial_state(), step_index=0)

    assert observation["current_macro"] == "u1"
    assert observation["current_macro_index"] == 0
    assert tuple(observation["canvas"].shape) == (6, 4, 4)
    assert tuple(observation["action_mask"].shape) == (4, 4)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/chip/test_rl_adapter.py -q`
Expected: FAIL because `observe()` does not yet expose PPO-facing fields.

- [ ] **Step 3: Implement the minimal adapter extensions**

```python
class ChipRLAdapter:
    def ordered_macros(self) -> tuple[str, ...]:
        movable = [macro for macro in self.problem.macros.values() if not macro.is_fixed]
        return tuple(macro.name for macro in sorted(movable, key=lambda macro: (-(macro.width * macro.height), macro.name)))
```

```python
def current_macro_for_step(self, step_index: int) -> str | None:
    ordered = self.ordered_macros()
    return ordered[step_index] if step_index < len(ordered) else None
```

```python
def observe(self, state: ChipRLState, step_index: int) -> dict[str, object]:
    return {
        "step_index": step_index,
        "current_macro": self.current_macro_for_step(step_index),
        "current_macro_index": step_index,
        "placed_count": step_index,
        "graph": self._build_graph(state, step_index=step_index),
        "canvas": self.build_canvas(state, step_index=step_index, grid_shape=self.grid_shape),
        "action_mask": self.build_action_mask(state, step_index=step_index, grid_shape=self.grid_shape),
        "placement": dict(state.placement),
        "movable_macros": self.movable_macros,
        "hpwl": self.current_hpwl(state),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/chip/test_rl_adapter.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/rl_adapter.py tests/chip/test_rl_adapter.py
git commit -m "feat: add ppo-facing chip rl observations"
```

### Task 2: Add The PPO Policy Model

**Files:**
- Create: `engine/ppo_policy.py`
- Create: `tests/engine/test_ppo_policy.py`
- Modify: `engine/__init__.py`

- [ ] **Step 1: Write the failing PPO policy tests**

```python
def test_ppo_policy_outputs_logits_and_value():
    model = PPOPlacementPolicy(node_feature_dim=4, graph_feature_dim=2, grid_channels=6, grid_shape=(4, 4), hidden_dim=16)
    logits, value = model(graph_batch=batch, canvas=canvas, current_macro_index=current_macro_index, action_mask=action_mask)

    assert tuple(logits.shape) == (2, 16)
    assert tuple(value.shape) == (2, 1)
```

```python
def test_ppo_policy_masks_invalid_actions_with_negative_infinity():
    masked_logits, _ = model(...)
    assert torch.isneginf(masked_logits[0, 0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/engine/test_ppo_policy.py -q`
Expected: FAIL because `engine/ppo_policy.py` does not exist.

- [ ] **Step 3: Implement the minimal PPO policy**

```python
class PPOPlacementPolicy(nn.Module):
    def forward(self, graph_batch, canvas, current_macro_index, action_mask):
        ...
        masked_logits = logits.masked_fill(action_mask.view(batch_size, -1), float("-inf"))
        return masked_logits, value
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/engine/test_ppo_policy.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/ppo_policy.py engine/__init__.py tests/engine/test_ppo_policy.py
git commit -m "feat: add phase3 ppo policy skeleton"
```

### Task 3: Add Masked Random And Greedy Bring-Up Baselines

**Files:**
- Create: `chip/experiments/phase3_baselines.py`
- Create: `tests/chip/test_phase3_baselines.py`

- [ ] **Step 1: Write the failing baseline tests**

```python
def test_masked_random_baseline_only_emits_valid_actions():
    result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=2, seed=7)
    assert result["invalid_action_count"] == 0
```

```python
def test_greedy_baseline_improves_over_random_on_tiny_problem():
    greedy = run_greedy_hpwl_baseline(problem=build_problem(), grid_shape=(4, 4))
    random_result = run_masked_random_baseline(problem=build_problem(), grid_shape=(4, 4), steps=1, seed=0)
    assert greedy["final_hpwl"] <= random_result["final_hpwl"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/chip/test_phase3_baselines.py -q`
Expected: FAIL because the baseline module does not exist.

- [ ] **Step 3: Implement the minimal baseline harness**

```python
def run_masked_random_baseline(...):
    ...
```

```python
def run_greedy_hpwl_baseline(...):
    ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/chip/test_phase3_baselines.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chip/experiments/phase3_baselines.py tests/chip/test_phase3_baselines.py
git commit -m "feat: add phase3 bring-up baselines"
```

### Task 4: Verify The Full Bring-Up Slice

**Files:**
- Modify: `docs/superpowers/task-memory.md`

- [ ] **Step 1: Run focused Phase 3 tests**

Run: `pytest tests/engine/test_rl_env.py tests/chip/test_rl_adapter.py tests/chip/test_phase3_random_agent.py tests/engine/test_ppo_policy.py tests/chip/test_phase3_baselines.py -q`
Expected: PASS

- [ ] **Step 2: Run the broader engine/chip suite**

Run: `pytest tests/engine tests/chip -q`
Expected: PASS

- [ ] **Step 3: Update task memory with Phase 3 PPO bring-up status**

```markdown
## 2026-04-07 Phase 3 PPO Bring-Up

- PPO-facing adapter observations exist.
- PPO policy skeleton emits masked logits and value estimates.
- Masked random and greedy HPWL baselines are implemented and green.
- PPO training loop is still open.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/task-memory.md
git commit -m "docs: record phase3 ppo bring-up status"
```
