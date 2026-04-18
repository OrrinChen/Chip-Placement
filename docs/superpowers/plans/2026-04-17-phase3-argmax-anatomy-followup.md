# Phase 3 Argmax Anatomy Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Ariane trajectory-space diagnostic so we can distinguish tie-breaking fragility from broader policy breakage.

**Architecture:** Reuse the fixed `pi_init` reference trace and the existing argmax-agreement helper. First add the `alpha/10` null-reference comparison, then enrich the agreement summary with distribution-shaped evidence (entropy/margin histograms and macro-size buckets), and only then run the real Ariane diagnostics.

**Tech Stack:** Python, pytest, existing Phase 3 PPO experiment harness, JSON artifacts

---

### Task 1: Add failing tests for richer argmax-agreement summaries

**Files:**
- Modify: `C:\Users\14574\Chip\tests\chip\test_phase3_ppo.py`

- [ ] Add a failing test that requires:
  - entropy/margin histogram or CDF-style bucket summaries for flipped vs non-flipped steps
  - macro-size bucket summaries keyed by macro area tier
- [ ] Run:
  - `python -m pytest tests/chip/test_phase3_ppo.py -k argmax_agreement -q`
- [ ] Verify the new test fails for the expected missing fields.

### Task 2: Implement minimal richer agreement summary

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_ppo.py`

- [ ] Add the minimal production code to make Task 1 pass.
- [ ] Keep behavior observational only; do not alter rollout or policy behavior.
- [ ] Re-run:
  - `python -m pytest tests/chip/test_phase3_ppo.py -k argmax_agreement -q`

### Task 3: Add failing integration test for alpha/10 null-reference diagnostic

**Files:**
- Modify: `C:\Users\14574\Chip\tests\chip\test_phase3_macroplacement_ppo.py`

- [ ] Add a failing test that requires the Ariane runner to:
  - compare baseline full-LR substep0 and `alpha/10` substep0 against the same fixed `pi_init` trace
  - persist both agreement summaries into one artifact
- [ ] Run:
  - `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k argmax_agreement -q`
- [ ] Verify the new test fails for the expected missing runner fields.

### Task 4: Implement minimal Ariane follow-up runner changes

**Files:**
- Modify: `C:\Users\14574\Chip\chip\experiments\phase3_macroplacement_ppo.py`

- [ ] Implement the smallest runner change needed for Task 3.
- [ ] Re-run:
  - `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k argmax_agreement -q`

### Task 5: Verify and run the real Ariane follow-up diagnostic

**Files:**
- Modify: `C:\Users\14574\Chip\docs\superpowers\task-memory.md`
- Create/update artifact directory under: `C:\Users\14574\Chip\experiments\chip\phase3_argmax_agreement_v1\`

- [ ] Run targeted verification:
  - `python -m pytest tests/chip/test_phase3_ppo.py -k argmax_agreement -q`
  - `python -m pytest tests/chip/test_phase3_macroplacement_ppo.py -k argmax_agreement -q`
- [ ] Run full verification:
  - `python -m pytest tests/engine tests/chip -q`
- [ ] Run the real Ariane diagnostic and save the artifact.
- [ ] Update task memory with:
  - `alpha/10` null-reference interpretation
  - distribution-shape observations
  - macro-size bucket observations
