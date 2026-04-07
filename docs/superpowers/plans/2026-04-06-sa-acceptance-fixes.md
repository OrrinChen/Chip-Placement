# SA Acceptance Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Phase 1 SA acceptance flow evaluate an independent, row-aware SA baseline instead of replaying the built-in placement.

**Architecture:** Keep the OpenROAD acceptance chain intact, but decouple SA from the built-in seed by deriving a separate analytical seed on the clean DEF. Add row metadata to `ChipProblem` so both SA neighbor generation and OpenROAD injection preserve legal row semantics in `x` and `y`.

**Tech Stack:** Python, pytest, OpenROAD Docker bridge, CVXPY analytical seed, simulated annealing

---

### Task 1: Lock the Regression Surface with Tests

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\tests\chip\test_phase1_acceptance.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\tests\chip\test_baseline_experiment.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\tests\chip\test_netlist_parser.py`

- [ ] Add a failing acceptance test that proves SA no longer loads `seed_placed.def` as its starting state and instead legalizes from the clean DEF.
- [ ] Add a failing placement injection test that expects `x` and `y` to snap to placement rows and to use the selected row orientation.
- [ ] Add a failing neighbor-generation test that expects vertical SA moves to land on row-aligned `y` coordinates instead of arbitrary `±5` micron offsets.
- [ ] Run `pytest tests/chip/test_phase1_acceptance.py tests/chip/test_baseline_experiment.py tests/chip/test_netlist_parser.py -v` and confirm the new assertions fail for the intended reasons.

### Task 2: Add Row-Aware Placement Primitives

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\chip\problem.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\chip\netlist_parser.py`

- [ ] Add a shared `PlacementRow` model to `chip.problem` and thread parsed rows through `ChipProblem`.
- [ ] Expose enough row metadata for downstream code to snap placements onto row origins and site pitches without reparsing DEF files.
- [ ] Run `pytest tests/chip/test_netlist_parser.py -v` and confirm the parser now preserves row data while existing parsing behavior stays green.

### Task 3: Decouple SA from the Built-In Seed

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\chip\experiments\phase1_acceptance.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\chip\experiments\baseline.py`
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\engine\analytical_solver.py` only if needed for seed compatibility

- [ ] Build an analytical SA seed from the clean DEF problem graph instead of the built-in placement.
- [ ] Make `_run_sa_baseline` accept an explicit initial placement so acceptance and future dataset code can control the seed.
- [ ] Update the SA path to legalize and evaluate against the clean DEF base, not `seed_placed.def`.
- [ ] Make `_neighbor` row-aware in both axes, preserving site-aligned `x` moves and row-height-aligned `y` moves.
- [ ] Run `pytest tests/chip/test_phase1_acceptance.py tests/chip/test_baseline_experiment.py -v` and confirm the bug-fix behavior is green.

### Task 4: Re-Verify and Sweep for a Sane Iteration Count

**Files:**
- Modify: `C:\Users\14574\Chip\.worktrees\phase2-foundation\chip\ispd_benchmarks.py` only if a new default iteration count is chosen

- [ ] Run `pytest tests/engine tests/chip -v` to verify the full focused suite after the behavior changes.
- [ ] Run a convergence sweep on `mgc_pci_bridge32_1` for `1000, 5000, 10000, 50000` iterations and capture objective/runtime trends before changing defaults.
- [ ] Choose a default SA iteration count based on the first clear plateau that still keeps `mgc_des_perf_2` practical to rerun.
- [ ] Decide whether the acceptance suite is ready for a full rerun with the new SA settings.
