# Project: Graph-Based Combinatorial Placement Engine

## Core Thesis

Build a domain-agnostic graph-based placement/scheduling engine, validated first on chip design (primary project, highest moat), architecturally ready to spawn a standalone AI infrastructure project (secondary project, largest market).

The two projects share a common engine but diverge on domain layer, constraints, evaluation, and deployment story. The engine is the reusable IP. The domain layers are the moat.

## Architecture: Three-Layer Separation

```
┌─────────────────────────────────────────────────────────┐
│              Domain Layer (swappable)                    │
│                                                         │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │   Chip Design     │    │   AI Infrastructure       │  │
│  │                   │    │                           │  │
│  │ netlist parser    │    │ request DAG builder       │  │
│  │ DEF/LEF reader    │    │ GPU topology model        │  │
│  │ timing extractor  │    │ KV cache state tracker    │  │
│  │ DRC rule engine   │    │ SLA constraint encoder    │  │
│  │ OpenROAD bridge   │    │ vLLM/SGLang bridge        │  │
│  └────────┬─────────┘    └────────────┬──────────────┘  │
│           │                           │                  │
│           ▼                           ▼                  │
├─────────────────────────────────────────────────────────┤
│              Engine Layer (shared)                       │
│                                                         │
│  graph_repr:   node/edge/hyperedge graph abstraction    │
│  gnn_proxy:    GNN-based quality predictor              │
│  or_solver:    analytical placement / ILP / SA          │
│  rl_agent:     PPO/SAC sequential placement agent       │
│  constraint:   generic constraint registry + checker    │
│                                                         │
├─────────────────────────────────────────────────────────┤
│              Evaluation Layer (per-domain)               │
│                                                         │
│  chip:  wire length, timing slack, congestion, DRC      │
│  infra: throughput, TTFT, TPOT, SLA violation, util     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The critical design rule: Engine Layer never imports from Domain Layer. Domain Layer calls Engine Layer APIs. This is what makes the infra side a genuine standalone project rather than a thin demo.

## What Makes Infra a Real Standalone Project (Not a Thin Adapter)

If the chip design project is the primary deliverable, the infra side needs its own genuine hard problems to be credible as independent work. These are the infra-specific components that don't exist in chip design:

- KV cache memory management: online paging problem with variable-size allocations, eviction under memory pressure, prefix sharing across requests. This is a genuine research problem with no good solution today.
- Continuous batching dynamics: requests enter and leave the batch at different times, each request's compute cost changes as it progresses through decode. The scheduling problem is fundamentally different from chip placement because it's online and non-preemptive.
- Speculative decoding orchestration: when to use draft model vs main model, how to schedule verification batches, how to handle rejection cascades. This is a multi-model scheduling problem.
- Multi-tier serving: routing between different model sizes/quantizations based on request complexity. Which requests can be served by a small model vs must go to the large model?

Each of these is a substantial engineering and research challenge. The shared engine handles the graph optimization and RL infrastructure; these components handle the infra-specific logic that makes it a real project.

## Project 1: AI for Chip Design (Primary)

### Learning Phase (Weeks 1-4)

Goal: understand chip design domain deeply enough to be productive. Do not write project code during this phase.

Week 1-2: OpenROAD flow walkthrough.
- Install OpenROAD, run a complete flow on a small ISPD benchmark (e.g., adaptec1 or ibm01).
- Trace the flow: read Verilog → synthesis → floorplan → placement → CTS → global route → detailed route → timing analysis.
- For each step, understand: what is the input, what is the output, what are the constraints, what is the objective.
- Key concepts to internalize: netlist, standard cell, macro, pin, net, die area, placement density, HPWL (half-perimeter wire length), timing path, slack, clock skew, congestion map, DRC violations.
- Deliverable: a markdown document mapping each concept to its analog in your existing knowledge (e.g., "placement density constraint ≈ position limit constraint in portfolio optimization").

Week 3: Study existing ML for EDA work.
- Read Google's Nature 2021 paper (chip placement with RL). Focus on: state representation, action space, reward design, training curriculum.
- Read the criticism papers (Cheng et al. 2023 rebuttal, ISPD 2023 contest results). Understand what the controversy is about: reproducibility, comparison fairness, scalability.
- Read DREAMPlace (analytical placement accelerated with GPU). Understand how analytical methods differ from RL-based methods.
- Skim Circuit Training (Google's open-source implementation). Understand the code structure.
- Deliverable: a 1-page note on what the current SOTA does well and where the gaps are.

Week 4: GNN fundamentals + PyTorch Geometric.
- Learn message passing, GCN, GAT, GraphSAGE.
- Do one tutorial: node classification on Cora dataset, graph classification on TU dataset.
- Understand how circuit netlists map to graphs: cells → nodes, nets → hyperedges (or clique expansion to edges), pin positions → node features.
- Deliverable: a small script that loads an ISPD benchmark netlist into a PyG graph object.

### Phase 1: Engine Foundation + OR Baseline (Weeks 5-7)

Build the domain-agnostic engine layer and the chip design domain layer simultaneously.

Engine layer:
- `engine/graph_repr.py`: generic graph with typed nodes, typed edges, node features, edge features. Backed by PyG Data object.
- `engine/constraint.py`: constraint registry. Each constraint is a callable that takes a placement and returns (satisfied: bool, violation_magnitude: float). Domain layer registers constraints; engine checks them.
- `engine/sa_solver.py`: simulated annealing placement. Domain-agnostic: it takes a graph, a cost function, constraints, and an initial placement. Returns best placement found.
- `engine/analytical_solver.py`: quadratic placement (minimize sum of squared wire lengths). Uses cvxpy. Gives a continuous relaxation that can be used as initial seed for SA or RL.

Chip design domain layer:
- `chip/netlist_parser.py`: reads DEF/LEF files from ISPD benchmark, produces engine graph.
- `chip/cost.py`: HPWL cost function (half-perimeter wire length — standard proxy for routed wire length).
- `chip/constraints.py`: boundary constraint (macros must be inside die area), non-overlap constraint, minimum spacing.
- `chip/openroad_bridge.py`: calls OpenROAD to do global routing + STA on a given placement, returns actual wire length and timing slack. This is the ground-truth evaluator.

Evaluation:
- Run SA on 3 ISPD benchmarks (small, medium, large).
- Compare SA placement quality (HPWL, routed wire length via OpenROAD) against OpenROAD's built-in placer.
- This establishes your optimization baseline.

Deliverables:
- Working SA placer on ISPD benchmarks.
- Baseline results table.
- Engine layer with clean API that chip domain layer uses but doesn't couple to.

### Phase 2: GNN Proxy Model (Weeks 8-9)

Goal: train a GNN that predicts placement quality without running expensive OpenROAD routing.

Data generation:
- For each ISPD benchmark, generate 500-1000 random/perturbed placements.
- For each placement, run OpenROAD global routing + STA.
- Record: HPWL, routed wire length, worst timing slack, congestion hotspot count.
- This creates a (placement_graph, quality_metrics) dataset.

Model:
- `engine/gnn_proxy.py`: domain-agnostic GNN. Input: graph with node features (position, size, type) and edge features (connectivity weight). Output: predicted quality score.
- Architecture: 3-4 layers of GraphSAGE or GAT, graph-level readout via mean pooling, MLP head predicting quality metrics.
- Training: MSE loss on normalized quality metrics. 80/20 train/test split across placements (not across benchmarks — test on held-out placements of the same benchmark).

Validation:
- Rank correlation between GNN-predicted quality and actual OpenROAD quality across held-out placements.
- Target: Spearman correlation > 0.8. If lower, the proxy is not accurate enough to guide RL.

Deliverables:
- Trained GNN proxy model.
- Correlation report on held-out placements.
- Speed comparison: GNN prediction (milliseconds) vs OpenROAD routing (minutes).

### Phase 3: RL Placement Agent (Weeks 10-12)

Goal: train an RL agent that sequentially places macros, using GNN proxy as the reward signal.

Environment:
- `engine/rl_env.py`: Gymnasium-compatible. Domain-agnostic.
- State: current partial placement encoded as a graph (GNN embedding), remaining items to place, canvas utilization map.
- Action: grid position for the next item. Canvas discretized into grid (e.g., 32x32 or 64x64).
- Reward: negative GNN-predicted cost of current placement. Terminal reward: negative actual cost from domain evaluator (OpenROAD for chip).
- Episode: one complete placement of all macros.

Chip-specific:
- `chip/rl_adapter.py`: translates chip-specific state (netlist, die area, placed macros) into engine-level graph state. Translates engine-level action (grid position) into physical coordinates respecting die boundary.

Training:
- PPO with GNN-based policy network.
- Curriculum: start with small benchmark (fewer macros), then scale to larger.
- Baseline comparison: RL vs SA vs analytical placement vs OpenROAD default.

Key risk: RL placement may not beat SA on all benchmarks. Mitigation: even if RL ties or slightly loses to SA, the GNN proxy model and the framework design are still valuable contributions. RL can be positioned as "competitive with SA but 10x faster at inference time" — speed is a legitimate advantage even if quality is comparable.

Deliverables:
- Trained RL placement agent.
- Results table: RL vs SA vs analytical vs OpenROAD on 3+ benchmarks.
- Metrics: HPWL, routed wire length, timing slack, runtime.
- Learning curves and training diagnostics.

### Phase 4: Formal Constraint Verification (Week 13)

Goal: add a formal constraint checking layer that verifies placements against design rules.

Implementation:
- `chip/drc_checker.py`: encodes design rules as formal constraints.
  - Minimum spacing between macros.
  - Macro-to-boundary minimum distance.
  - Pin accessibility (no macro blocks another macro's pin access).
  - Power/ground strap clearance.
- `engine/constraint_verifier.py`: generic constraint verifier. Takes a placement and a set of registered constraints, returns (pass/fail, list of violations with locations and severity).

Integration:
- RL agent: add DRC violation count as a penalty term in reward.
- SA solver: reject moves that create DRC violations.
- Post-placement: run full DRC check and report.

This connects your MIE1501 formal reasoning training to a practical application.

Deliverables:
- DRC checker with 5-10 rules.
- Before/after comparison: placement quality with and without DRC-aware optimization.
- Violation report format.

### Phase 5: Evaluation and Report (Week 14)

Goal: comprehensive evaluation on ISPD benchmarks + project documentation.

Evaluation:
- Run all methods (SA, analytical, RL, RL+DRC) on 3-5 ISPD benchmarks.
- Full OpenROAD flow on each final placement: global route, detailed route, STA, DRC.
- Report: HPWL, routed WL, worst negative slack, total negative slack, congestion, DRC violations, runtime.
- Ablation: GNN proxy accuracy vs RL performance (does better proxy → better RL?).

Documentation:
- README with project overview, architecture diagram, results summary.
- Reproducibility instructions.
- Interactive visualization: placement canvas showing macro positions, congestion heatmap, timing-critical paths highlighted.

## Project 2: AI Infrastructure Scheduling (Standalone Extension)

This is NOT a thin demo adapter. It is a genuine standalone project that shares the engine layer but has its own hard problems, its own data, and its own evaluation.

### When to Start

Only after chip design Phase 3 is complete and the engine layer API is stable. Starting infra work before the engine is solid will lead to premature abstraction.

### Phase I.1: GPU Simulator + Workload Model (2-3 weeks)

- `infra/gpu_sim.py`: discrete-event simulator for a GPU cluster.
  - Models: GPU memory (HBM), compute throughput (tokens/sec vs batch size curve), KV cache allocation/eviction, network bandwidth between GPUs.
  - Calibrated from public benchmarks (A100/H100 throughput curves).
- `infra/workload.py`: workload generator from Azure LLM Inference Trace.
  - Replays real production request patterns.
  - Each request: arrival time, prompt length, expected completion length, SLA class.
- `infra/request_dag.py`: translates request stream into engine-compatible graph.
  - Nodes: individual requests.
  - Edges: resource contention (requests competing for same GPU), dependency (multi-step agent calls).
  - Node features: prompt length, priority, arrival time.
  - Constraints: GPU memory capacity, SLA deadline, KV cache budget.

### Phase I.2: KV Cache Management as Combinatorial Problem (2 weeks)

This is the infra-specific hard problem that has no analog in chip design.

- `infra/kv_cache_manager.py`: online paging with variable-size pages.
  - Each in-flight request holds KV cache proportional to its current sequence length.
  - When memory pressure hits, must decide which request's cache to evict (request paused, cache recomputed later).
  - Eviction decision is a combinatorial problem: minimize total recomputation cost subject to memory budget.
  - Use OR solver for offline optimal (given full future knowledge), RL for online adaptive.
- Prefix sharing: requests with common prompt prefixes can share KV cache pages. Detecting and exploiting this is a graph matching problem.

### Phase I.3: RL Scheduling Agent (2-3 weeks)

- Reuse `engine/rl_env.py` with infra-specific adapter.
- State: GPU utilization, pending request queue, in-flight request states, KV cache occupancy, recent arrival rate.
- Action: for each pending request, assign to GPU and batch position; for KV cache pressure, select eviction candidates.
- Reward: negative (mean TTFT + mean TPOT) weighted by SLA priority, minus recomputation cost from evictions.
- Compare: RL vs FCFS vs shortest-job-first vs vLLM-style continuous batching.

### Phase I.4: Evaluation (1 week)

- Run on Azure LLM Inference Trace.
- Metrics: throughput (tokens/s), P50/P95/P99 TTFT, P50/P95/P99 TPOT, SLA violation rate, GPU utilization, KV cache hit rate.
- Baselines: FCFS, SJF, round-robin, simulated vLLM scheduling.
- Ablation: OR-only vs RL-only vs OR+RL.

### Total Infra Timeline: 7-9 weeks (standalone, after engine is stable)

## Shared Engine Layer API Contract

This is what makes the two projects genuinely share code rather than being copy-paste.

```python
# engine/graph_repr.py
class PlacementGraph:
    """Domain-agnostic graph representation."""
    nodes: List[Node]           # items to place/schedule
    edges: List[Edge]           # connectivity/dependency
    canvas: Canvas              # 2D grid (chip) or resource slots (GPU)

# engine/constraint.py
class ConstraintRegistry:
    """Domain registers constraints, engine checks them."""
    def register(name, check_fn, penalty_weight): ...
    def check_all(placement) -> ConstraintReport: ...

# engine/gnn_proxy.py
class GNNProxy:
    """Predicts placement quality from graph."""
    def predict(graph, placement) -> QualityScore: ...
    def train(dataset: List[Tuple[Placement, GroundTruthMetrics]]): ...

# engine/rl_env.py
class PlacementEnv(gymnasium.Env):
    """Sequential placement environment."""
    def __init__(graph, constraints, proxy, domain_evaluator): ...
    def step(action) -> (obs, reward, done, info): ...

# engine/sa_solver.py
class SAPlacement:
    """Simulated annealing baseline."""
    def solve(graph, cost_fn, constraints, ...) -> Placement: ...
```

Domain layers implement:
```python
# Domain adapter interface
class DomainAdapter:
    def load_problem(path) -> PlacementGraph: ...
    def translate_action(engine_action) -> domain_action: ...
    def evaluate_placement(placement) -> DomainMetrics: ...
    def get_constraints() -> List[Constraint]: ...
```

## File Structure

```
graph-placement-engine/
├── engine/                         # domain-agnostic (shared IP)
│   ├── graph_repr.py
│   ├── constraint.py
│   ├── gnn_proxy.py
│   ├── rl_env.py
│   ├── rl_agent.py                 # PPO/SAC training
│   ├── sa_solver.py
│   ├── analytical_solver.py
│   └── tests/
├── chip/                           # chip design domain
│   ├── netlist_parser.py
│   ├── cost.py
│   ├── constraints.py
│   ├── drc_checker.py
│   ├── openroad_bridge.py
│   ├── rl_adapter.py
│   ├── benchmarks/                 # ISPD benchmark configs
│   └── tests/
├── infra/                          # AI infrastructure domain (future standalone)
│   ├── gpu_sim.py
│   ├── workload.py
│   ├── request_dag.py
│   ├── kv_cache_manager.py
│   ├── scheduling_adapter.py
│   ├── traces/                     # Azure trace data
│   └── tests/
├── experiments/
│   ├── chip/                       # chip experiment scripts + results
│   └── infra/                      # infra experiment scripts + results
├── models/                         # trained model checkpoints
├── docs/
│   ├── chip_design_primer.md       # domain knowledge notes
│   ├── architecture.md             # engine design doc
│   └── results/                    # evaluation reports
└── README.md
```

When infra grows into standalone project, it can be extracted into its own repo with engine/ as a git submodule or package dependency.

## Timeline Summary

| Phase | Weeks | Cumulative |
|-------|-------|------------|
| Learning (chip domain + GNN) | 4 | 4 |
| Phase 1: Engine + OR baseline | 3 | 7 |
| Phase 2: GNN proxy | 2 | 9 |
| Phase 3: RL placement | 3 | 12 |
| Phase 4: Formal DRC | 1 | 13 |
| Phase 5: Evaluation + report | 1 | 14 |
| --- Chip design complete --- | | |
| Phase I.1: GPU sim + workload | 2-3 | 16-17 |
| Phase I.2: KV cache management | 2 | 18-19 |
| Phase I.3: RL scheduling | 2-3 | 20-22 |
| Phase I.4: Infra evaluation | 1 | 21-23 |

Chip design alone: ~14 weeks.
Chip design + standalone infra: ~21-23 weeks.
The two can overlap with PortfolioOS work if needed.

## Interview Narratives

### For semiconductor companies (NVIDIA, Tenstorrent, AMD, Synopsys, Cadence)

"I built a graph-based placement engine for chip design. It uses GNN proxy models to accelerate placement evaluation by 1000x compared to running full routing, and an RL agent that sequentially places macros while respecting design rules verified by a formal constraint checker. On ISPD benchmarks, it achieves comparable wire length to simulated annealing in 1/10th the runtime."

### For AI infrastructure companies (AWS, Google Cloud, Anthropic, OpenAI)

"I built a domain-agnostic graph-based scheduling engine and validated it on two very different problems: chip placement and LLM inference serving. For inference, it handles KV cache management as an online combinatorial optimization problem, uses RL for adaptive request routing, and reduces P99 latency by X% compared to FCFS scheduling on real Azure production traces."

### For general ML/RL roles

"I specialize in applying RL and combinatorial optimization to real-world sequential decision problems. I've built production-grade systems in three domains: financial portfolio management, semiconductor chip design, and AI infrastructure. The common thread is: predict → optimize → adapt → evaluate, with formal constraint verification at each step."

Three projects, one methodology, three domains. This is the profile of someone with a transferable problem-solving framework, not someone who learned one trick in one domain.
