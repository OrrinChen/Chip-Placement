# Phase 2 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the roadmap Phase 2 loop by generating OpenROAD-scored placement samples, training a GNN proxy on those samples, and emitting validation and speed reports.

**Architecture:** Keep data generation in `chip/experiments/phase2_dataset.py`, keep the proxy model in `engine/gnn_proxy.py`, and put chip-specific dataset conversion and training orchestration in `chip/experiments/phase2_proxy.py`. Long OpenROAD collection runs should be resumable and failure-aware, while proxy training should consume materialized JSON rows rather than invoking OpenROAD directly.

**Tech Stack:** Python 3.11, pytest, PyTorch, NumPy, SciPy, Dockerized OpenROAD, JSON artifacts

---

## File Map

- Modify: `chip/experiments/phase2_dataset.py`
- Modify: `chip/openroad_eval.py`
- Modify: `chip/experiments/phase1_acceptance.py`
- Create: `engine/gnn_proxy.py`
- Modify: `engine/__init__.py`
- Create: `tests/engine/test_gnn_proxy.py`
- Create: `chip/experiments/phase2_proxy.py`
- Modify: `tests/chip/test_phase2_dataset.py`
- Create: `tests/chip/test_phase2_proxy.py`
- Modify: `docs/superpowers/task-memory.md`

### Task 1: Finish Dataset Collection

**Files:**
- Modify: `chip/experiments/phase2_dataset.py`
- Modify: `chip/openroad_eval.py`
- Modify: `tests/chip/test_phase2_dataset.py`

- [ ] Add failing tests for materialized ground-truth collection, including success and timeout rows.
- [ ] Implement per-sample artifact paths, row persistence, and batch collection entrypoints.
- [ ] Add sample synthesis for `random_perturbation` and `sa_truncated` placements using the existing SA/legalization helpers.
- [ ] Thread `timeout_seconds` through OpenROAD evaluation so large batches can fail fast per sample.
- [ ] Run `pytest tests/chip/test_phase2_dataset.py tests/chip/test_openroad_eval.py -v`.

### Task 2: Add The Domain-Agnostic GNN Proxy

**Files:**
- Create: `engine/gnn_proxy.py`
- Modify: `engine/__init__.py`
- Create: `tests/engine/test_gnn_proxy.py`

- [ ] Add failing tests for graph batching, forward-pass shape, and a tiny overfit case.
- [ ] Implement a lightweight GraphSAGE-style proxy in pure PyTorch over `ProblemGraph`.
- [ ] Expose graph encoding, pooled prediction, and normalized regression outputs.
- [ ] Run `pytest tests/engine/test_gnn_proxy.py -v`.

### Task 3: Build Phase 2 Training And Reporting

**Files:**
- Create: `chip/experiments/phase2_proxy.py`
- Create: `tests/chip/test_phase2_proxy.py`

- [ ] Add failing tests for loading JSON rows into graph/target tensors and for validation summary generation.
- [ ] Implement dataset loading, train/validation split, normalization, training loop, Spearman calculation, and report emission.
- [ ] Emit model checkpoint path, correlation report, and speed comparison summary.
- [ ] Run `pytest tests/chip/test_phase2_proxy.py -v`.

### Task 4: Verify And Launch Overnight Runs

**Files:**
- Modify: `docs/superpowers/task-memory.md`

- [ ] Run `pytest tests/engine tests/chip -v`.
- [ ] Launch a resumable Phase 2 collection batch in the background from the worktree.
- [ ] Launch proxy training against any collected rows that are already available.
- [ ] Update task memory with run commands, artifact locations, and current batch state.
