# Graph Placement Engine

Graph Placement Engine is a research-first codebase for graph-based placement optimization, with a current focus on chip design.

The repository is organized around a shared optimization layer in `engine/` and a chip-design domain layer in `chip/`. It currently covers three major workstreams:

- `Phase 1`: engine foundations plus simulated-annealing / OpenROAD acceptance on ISPD-style benchmarks
- `Phase 2`: stable dataset generation plus a production proxy model for placement-quality prediction
- `Phase 3`: search / imitation / PPO experiments, first on toy benchmark ladders and then on real macro-placement benchmarks such as Ariane

This is not yet a polished package release. It is a reproducible experiment repo with committed artifacts, CLI entrypoints, and tests.

## Current Status

| Phase | Scope | Status on `main` | Primary artifact |
| --- | --- | --- | --- |
| Phase 1 | SA baseline + OpenROAD acceptance on `mgc_*` benchmarks | Complete | `experiments/chip/phase1_acceptance_results.json` |
| Phase 2 | Stratified dataset + proxy model | Complete; production path is frozen linear base, GNN line parked | `experiments/chip/phase2_proxy_frozen_base_alpha0_full/phase2_proxy_report.json` |
| Phase 3 | Search / imitation / PPO | Partial; toy ladder is closed, Ariane macro-placement bring-up and PPO diagnostics are ongoing | `experiments/chip/phase3_final_report.json` |

Important current conclusions:

- `Phase 1` passes on three ISPD-style benchmarks and establishes a working SA/OpenROAD baseline.
- `Phase 2` dataset generation is closed:
  - `300/300` successful stratified samples
  - `0` timeouts
  - `0` errors
- `Phase 2` proxy gate is passed with the frozen linear base:
  - same-design mean Spearman: `0.9008`
  - cross-design mean Spearman: `0.3859`
  - `phase3_foundation_ready = true`
- `Phase 3` is still open:
  - toy benchmark ladder and teacher-strength sweep are committed
  - real-chip Ariane macro-placement feasibility and ORFS completion are committed
  - the full final comparison table across RL / SA / analytical / OpenROAD on `3+` real macro-placement benchmarks is not closed yet

## Repository Layout

```text
chip/
  constraints.py              Chip-specific constraints and legality
  cost.py                     HPWL and placement-quality helpers
  netlist_parser.py           LEF/DEF -> in-memory chip problem
  openroad_bridge.py          OpenROAD / ORFS bridge
  rl_adapter.py               Chip-specific RL state/action adapter
  experiments/                Phase 1/2/3 experiment entrypoints

engine/
  graph.py                    Shared graph representation
  constraints.py              Generic constraint reporting
  analytical_solver.py        Analytical baseline
  sa_solver.py                Simulated annealing baseline
  gnn_proxy.py                Proxy-model training code
  ppo_policy.py               PPO policy / training utilities
  rl_env.py                   Shared sequential placement environment

experiments/chip/
  *.json                      Committed reports and experiment artifacts

tests/
  engine/                     Shared engine tests
  chip/                       Chip-domain and experiment tests

docs/superpowers/
  task-memory.md              Running research log / current conclusions
  plans/                      Execution plans
  specs/                      Design notes and phase specs
```

## Requirements

### Python

- Python `3.11+`
- The repository currently declares only minimal packaging metadata in `pyproject.toml`
- In practice, you will also need at least:
  - `torch`
  - `scipy`
  - `cvxpy`
  - `pytest`

### External tools

Depending on which phase you want to reproduce, you may also need:

- benchmark assets under `tmp/`
- Docker access to `openroad/orfs` for OpenROAD-backed evaluation and ORFS completion
- a local checkout of the TILOS MacroPlacement assets under `tmp/MacroPlacement`
- the `Plc_client` extension used by some macro-placement utilities

### Platform note

- macOS / MacBook is fine for pure Python work:
  - running tests
  - Phase 2 diagnostics
  - most Phase 3 PPO / warm-start / critic diagnostics
- Linux / WSL is strongly preferred for:
  - OpenROAD-backed evaluation
  - ORFS completion
  - Docker-based real-chip flow work

## Expected Asset Layout

The repo assumes benchmark assets live outside the tracked source tree, under `tmp/`.

For `Phase 1` and `Phase 2`, the expected shape is roughly:

```text
tmp/
  mgc_pci_bridge32_1/
  mgc_pci_bridge32_2/
  mgc_des_perf_2/
  ispd2012/
```

For `Phase 3` real macro-placement, the code expects:

```text
tmp/
  MacroPlacement/
```

The Ariane resolver in `chip/experiments/phase3_macroplacement_benchmarks.py` looks for `tmp/MacroPlacement` automatically.

## Quickstart

### 1. Create an environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install torch scipy cvxpy pytest
```

### 2. Run the core test suite

```powershell
python -m pytest tests/engine tests/chip -q
```

At the current `main` checkpoint, this should pass cleanly.

### 3. Read the current state before running heavy experiments

Start with:

- `graph_placement_engine_roadmap.md`
- `docs/superpowers/task-memory.md`
- `experiments/chip/phase3_final_report.json`

Those three files tell you what the project is trying to do, what has already been closed, and what is still open.

## Reproducing Key Results

### Phase 1: Acceptance Baseline

Run the Phase 1 acceptance suite:

```powershell
python -m chip.experiments.phase1_acceptance
```

Default output:

- `experiments/chip/phase1_acceptance_results.json`

This suite compares the SA baseline against OpenROAD-backed evaluation on the three `mgc_*` acceptance benchmarks.

### Phase 2: Dataset and Proxy

Inspect dataset CLI options:

```powershell
python -m chip.experiments.phase2_dataset --help
```

Inspect proxy CLI options:

```powershell
python -m chip.experiments.phase2_proxy --help
```

Committed Phase 2 outputs:

- stable dataset summary:
  - `experiments/chip/phase2_dataset_full_stratified_main_summary.json`
- production proxy report:
  - `experiments/chip/phase2_proxy_frozen_base_alpha0_full/phase2_proxy_report.json`

The production Phase 2 path is the frozen linear base. The higher-capacity GNN/MLP line was investigated and explicitly parked.

### Phase 3A: Toy Benchmark Ladder

The toy ladder and teacher-strength sweep are already committed in:

- `experiments/chip/phase3_benchmark_ladder/summary.json`
- `experiments/chip/phase3_teacher_strength_sweep/summary.json`
- `experiments/chip/phase3_final_report.json`

These results support the sample-efficiency story for warm-started PPO on `tiny / small / medium`, but they are not the final real-chip evaluation.

### Phase 3B: Ariane Macro-Placement Bring-Up

Run the Ariane baseline bring-up:

```powershell
python -m chip.experiments.phase3_macroplacement_bringup --run all --output-dir experiments/chip/phase3_macroplacement_bringup
```

Run PPO sanity checks:

```powershell
python -m chip.experiments.phase3_macroplacement_ppo --run plumbing --updates 3 --output-dir experiments/chip/phase3_macroplacement_bringup
python -m chip.experiments.phase3_macroplacement_ppo --run overfit --updates 10 --output-dir experiments/chip/phase3_macroplacement_bringup
```

Key committed Ariane artifacts:

- macro-placement feasibility / gate status:
  - `experiments/chip/phase3_macroplacement_probe/ariane133_ng45_status.json`
- ORFS reference completion:
  - `experiments/chip/phase3_macroplacement_probe/ariane133_ng45_reference_orfs_completion.json`
- warm-start and PPO diagnostics:
  - `experiments/chip/phase3_warm_start_probe_50/phase3_ariane_warm_start_probe_recomputed_full.json`
  - `experiments/chip/phase3_warm_start_substep_diagnostic_epoch1/phase3_ariane_warm_start_substep_diagnostic.json`
  - `experiments/chip/phase3_critic_prefit_comparison_v1/phase3_ariane_critic_prefit_comparison.json`

Current Ariane facts on `main`:

- legality check is fast enough for PPO bring-up
- action-mask build is fast enough for PPO bring-up
- ORFS reference completion is clean (`drc_violations = 0`)
- ORFS completion is too slow for PPO inner-loop reward and is currently used for offline evaluation only
- PPO training dynamics on Ariane are still under active diagnosis

## Known Limits

- This is still a research repo, not a fully packaged framework.
- `pyproject.toml` does not yet pin the full runtime dependency set.
- Benchmark assets are not vendored into the repository.
- `Phase 3` is not closed:
  - the unified final table across RL / SA / analytical / OpenROAD on `3+` real macro-placement benchmarks is still missing
  - routed WL / timing / runtime are not yet unified into a final real-chip evaluation report
- `Phase 2` GNN/MLP proxy work is not the current production path:
  - the accepted path is the frozen linear base
  - the high-capacity learned head line is parked until a larger data regime is available

## Recommended Entry Path For New Contributors

If you are new to the repo, the least painful order is:

1. Run `python -m pytest tests/engine tests/chip -q`
2. Read `docs/superpowers/task-memory.md`
3. Read `experiments/chip/phase2_proxy_frozen_base_alpha0_full/phase2_proxy_report.json`
4. Read `experiments/chip/phase3_final_report.json`
5. Re-run `Phase 1` or the Ariane bring-up before touching heavier `Phase 3` PPO work

That sequence gets you from stable foundations to the currently open frontier without having to reverse-engineer old branches.

## Further Reading

- roadmap:
  - `graph_placement_engine_roadmap.md`
- running memory / latest conclusions:
  - `docs/superpowers/task-memory.md`
- design notes:
  - `docs/superpowers/specs/`
- execution plans:
  - `docs/superpowers/plans/`
