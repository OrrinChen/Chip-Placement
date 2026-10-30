# Masked PPO vs simulated annealing

**Finding.** On the in-repo toy ladder (tiny, small, and medium; seeds 0–4; analytic HPWL), simulated annealing beat masked PPO on every problem. SA's HPWL is also lower on every individual seed (15 pairs in each budget). A 5× longer PPO budget did not help.

The second run uses 40 PPO updates instead of 8 and a wider network (hidden dim 64 instead of 32). Masked PPO's mean HPWL went from 5.2 to 6.4 on tiny, from 73.36 to 95.32 on small, and from 102.6 to 109 on medium. That run also gave SA 2000 iterations instead of 200. SA's mean stayed 1 ± 0 on tiny and 23 ± 2.73861 on small, and moved from 66.9 ± 3.17017 to 60.5 ± 0.707107 on medium.

Both budgets were run on Python 3.12.3, torch 2.15.0+cpu, Linux. Lower analytic HPWL is better. The spread is the sample standard deviation across seeds 0–4 (`statistics.stdev`, ddof=1). Display strings use Python format `.6g`. Full precision is in the summary files. Every seed of both methods was legal (5/5).

Masked PPO is `PPOPlacementPolicy` trained from scratch. Illegal grid cells are masked with `-inf`. The reported HPWL is a deterministic argmax rollout after training, not the best stochastic training episode. Simulated annealing uses the Phase 1 neighbor in `chip/experiments/baseline.py`. Its search objective is analytic HPWL plus constraint penalty. The reported score is analytic HPWL of the incumbent. PPO places macros one at a time onto a legality mask. SA perturbs a full placement.

These runs are `tiny`, `small`, and `medium` in `chip/experiments/phase3_benchmarks.py`. They are separate from the Phase 2 Spearman results and from the Ariane reference-flow DRC count. They are not routed wirelength.

## toy_ladder

Config `experiments/ppo_vs_sa/configs/toy_ladder.json`: hidden dim 32, 8 PPO updates, learning rate `1e-3`, clip 0.2, gamma 0.99, entropy coef 0.01, value coef 0.5. SA uses 200 iterations, initial temperature 10, cooling 0.95.

Initial analytic HPWL: tiny 8, small 100, medium 80.5. Movable macro counts: 1, 3, and 8.

| Benchmark | Masked PPO | Simulated annealing | Paired delta (PPO − SA) |
| --- | --- | --- | --- |
| tiny | 5.2 ± 2.28035 | 1 ± 0 | 4.2 ± 2.28035 |
| small | 73.36 ± 11.7392 | 23 ± 2.73861 | 50.36 ± 13.6568 |
| medium | 102.6 ± 21.5389 | 66.9 ± 3.17017 | 35.7 ± 22.1912 |

On medium, masked PPO's mean (102.6) is worse than the initial placement (80.5). SA's mean (66.9) is better than that initial placement.

Summary: `experiments/ppo_vs_sa/runs/toy_ladder/summary.json`.

## toy_ladder_extended

Config `experiments/ppo_vs_sa/configs/toy_ladder_extended.json`: hidden dim 64, 40 PPO updates, the same other PPO hyperparameters, SA 2000 iterations, the same temperature schedule, the same five seeds.

| Benchmark | Masked PPO | Simulated annealing | Paired delta (PPO − SA) |
| --- | --- | --- | --- |
| tiny | 6.4 ± 2.60768 | 1 ± 0 | 5.4 ± 2.60768 |
| small | 95.32 ± 60.7031 | 23 ± 2.73861 | 72.32 ± 59.7894 |
| medium | 109 ± 22.5333 | 60.5 ± 0.707107 | 48.5 ± 22.4944 |

Small's PPO standard deviation is large because seed 1's deterministic eval HPWL is 203.2 while the other four seeds are 73.6, 70.0, 73.2, and 56.6.

Summary: `experiments/ppo_vs_sa/runs/toy_ladder_extended/summary.json`.

## What this implies

At this scale, classical simulated annealing is a strong baseline. SA improved the initial placement on every problem. Masked PPO's medium mean is worse than the initial placement in both budgets. A later claim that this masked PPO recipe is useful on these three problems would have to beat the SA numbers above on the same analytic-HPWL metric and the same seeds. The Phase 2 Spearman figures and the Ariane reference-flow 0 DRC count do not change this comparison.

## Hypotheses

These are hypotheses about the runs above. They are not extra measurements.

- **One-shot placement versus repeated local search.** PPO places each movable macro once. The episode length is the movable count (1, 3, or 8), and the reported score is one deterministic rollout. SA keeps an incumbent and applies 200 or 2000 full-placement moves. On the extended run, seed 0 of tiny keeps the fixed cell at (8, 0). SA moves the free cell from (0, 0) to (7, 0). PPO leaves it at (2, 2). Those coordinates are in `experiments/ppo_vs_sa/runs/toy_ladder_extended/placements.json`. That seed is not the five-seed mean.
- **These extra PPO updates did not settle the eval policy.** Means rose on all three problems when updates went from 8 to 40 and hidden dim from 32 to 64. Extended small seed 1 evaluates at 203.2 against 73.6, 70.0, 73.2, and 56.6 on the other seeds. That describes this from-scratch recipe scored by a deterministic rollout. A warm start, another reward, and another algorithm were not run.
- **The neighborhood is cheap on these instances.** The slowest stored SA mean runtime is medium in the extended run, `0.0380016 ± 0.00127223` seconds. The other SA means in the two summaries are lower. The Ariane probe records 133 movable macros, and the ORFS completion record has 132 macros. This toy comparison does not include that netlist.

## Larger scale

An Ariane masked-PPO versus SA run has not been done. The checked-in config asks for `ariane133`, seeds 0–4, hidden dim 64, 50 PPO updates, and 500 SA iterations. The runner rejects that benchmark and writes no metrics:

```bash
python -m chip.experiments.ppo_vs_sa --config experiments/ppo_vs_sa/configs/ariane.json --output-dir experiments/ppo_vs_sa/runs/ariane
```

The command exits 2. A real run needs `tmp/MacroPlacement`. This entrypoint would still not report routed wirelength or DRC. The existing Ariane 0 DRC figure is the reference-floorplan ORFS completion, marked in that artifact as offline evaluation only. The toy-ladder result is not a prediction of the Ariane run.
