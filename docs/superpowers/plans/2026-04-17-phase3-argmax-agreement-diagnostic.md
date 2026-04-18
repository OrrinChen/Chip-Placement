# Phase 3 Argmax Agreement Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trajectory-space diagnostic that measures how much baseline first-substep PPO drift changes deterministic action choices on a fixed warm-start reference trace.

**Architecture:** Reuse the existing Phase 3 PPO rollout/action-trace plumbing. Add a small helper that compares `pi_init` against `pi_after_substep0` on an identical fixed state trace using argmax-vs-argmax decisions, then surface a minimal Ariane artifact for the baseline `1e-4` destructive run.

**Tech Stack:** Python 3.11, PyTorch, existing `chip.experiments.phase3_ppo` / `chip.experiments.phase3_macroplacement_ppo` utilities, pytest.

---

### Task 1: Add a focused agreement helper with TDD

**Files:**
- Modify: `C:\Users\14574\Chip\tests\chip\test_phase3_ppo.py`
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_ppo.py`

- [ ] Write a failing unit test for fixed-trace argmax agreement.
- [ ] Run the targeted pytest and verify it fails for the missing helper.
- [ ] Implement the minimal helper in `phase3_ppo.py`.
- [ ] Run the targeted pytest and verify it passes.

### Task 2: Add a thin Ariane runner for the baseline diagnostic

**Files:**
- Modify: `C:\Users\14574\Chip\tests\chip\test_phase3_macroplacement_ppo.py`
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_macroplacement_ppo.py`

- [ ] Write a failing integration-style test that checks baseline run wiring and artifact fields.
- [ ] Run the targeted pytest and verify it fails.
- [ ] Implement the minimal Ariane diagnostic runner using the new helper.
- [ ] Run the targeted pytest and verify it passes.

### Task 3: Run the real Ariane baseline diagnostic

**Files:**
- Output: `C:\Users\14574\Chip\experiments\chip\phase3_argmax_agreement_v1\*.json`
- Modify: `C:\Users\14574\Chip\docs\superpowers\task-memory.md`

- [ ] Run the baseline `1e-4` / fixed `pi_init` trace / argmax-vs-argmax diagnostic on Ariane.
- [ ] Inspect the resulting agreement rate before deciding whether deeper flip anatomy is warranted.
- [ ] Record the result and interpretation in task memory.
