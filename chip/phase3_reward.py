from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from chip.cost import hpwl_cost
from chip.experiments.phase2_dataset import summarize_displacement_distribution
from chip.experiments.phase2_proxy import fit_graph_stats_linear_model
from chip.experiments.phase2_proxy import predict_graph_stats_linear_model
from chip.problem import ChipProblem


@dataclass(frozen=True)
class RewardWeights:
    w_proxy: float = 1.0
    w_hpwl: float = 0.1
    legality_penalty: float = 10.0
    completion_bonus: float = 0.0


@dataclass
class EpisodeRewardNormalizer:
    epsilon: float = 1e-6
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def normalize(self, value: float) -> float:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.count < 2:
            scale = 1.0
        else:
            variance = self.m2 / max(self.count - 1, 1)
            scale = max(math.sqrt(max(variance, 0.0)), self.epsilon)
        return value / scale


@dataclass(frozen=True)
class FrozenLinearGraphStatsProxy:
    model: dict[str, torch.Tensor]
    target_name: str = "total_routed_wirelength"
    source_path: str | None = None

    @classmethod
    def from_phase2_dataset(
        cls,
        dataset_path: str | Path | None = None,
        *,
        target_name: str = "total_routed_wirelength",
    ) -> FrozenLinearGraphStatsProxy:
        resolved_dataset_path = Path(dataset_path) if dataset_path is not None else _default_phase2_dataset_path()
        rows = json.loads(resolved_dataset_path.read_text(encoding="utf-8-sig"))
        successful_rows = [row for row in rows if str(row.get("status", "success")) == "success" and target_name in row]
        targets = torch.tensor([[float(row[target_name])] for row in successful_rows], dtype=torch.float32)
        model = fit_graph_stats_linear_model(successful_rows, targets)
        return cls(model=model, target_name=target_name, source_path=str(resolved_dataset_path))

    def feature_row_for_placement(
        self,
        *,
        problem: ChipProblem,
        placement: dict[str, tuple[float, float]],
        seed_placement: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        return summarize_displacement_distribution(
            problem=problem,
            baseline_placement=seed_placement,
            candidate_placement=placement,
        )

    def predict_cost(
        self,
        *,
        problem: ChipProblem,
        placement: dict[str, tuple[float, float]],
        seed_placement: dict[str, tuple[float, float]],
    ) -> float:
        row = self.feature_row_for_placement(problem=problem, placement=placement, seed_placement=seed_placement)
        return float(predict_graph_stats_linear_model(self.model, [row]).squeeze().item())


@dataclass
class SeedCompletedRewardModel:
    problem: ChipProblem
    seed_placement: dict[str, tuple[float, float]]
    proxy: Any
    hpwl_evaluator: Any | None = None
    weights: RewardWeights = field(default_factory=RewardWeights)
    normalize_rewards: bool = True
    normalizer: EpisodeRewardNormalizer = field(default_factory=EpisodeRewardNormalizer)
    movable_macro_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.movable_macro_count = sum(1 for macro in self.problem.macros.values() if not macro.is_fixed)

    def reset_episode(self) -> None:
        self.normalizer.reset()

    def seed_complete(
        self,
        placement: dict[str, tuple[float, float]],
        *,
        placed_macros: frozenset[str] | set[str] | None = None,
    ) -> dict[str, tuple[float, float]]:
        completed = dict(self.seed_placement)
        if placed_macros is None:
            completed.update(placement)
            return completed
        placed_set = set(placed_macros)
        for macro_name, coords in placement.items():
            macro = self.problem.macros.get(macro_name)
            if macro is None or macro.is_fixed or macro_name in placed_set or macro_name not in completed:
                completed[macro_name] = coords
        return completed

    def compute_raw_step_reward(
        self,
        previous_state: Any,
        next_state: Any,
        *,
        is_legal: bool,
    ) -> float:
        previous_placement = self.seed_complete(previous_state.placement, placed_macros=getattr(previous_state, "placed_macros", None))
        next_placement = self.seed_complete(next_state.placement, placed_macros=getattr(next_state, "placed_macros", None))
        previous_proxy_cost = self.proxy.predict_cost(
            problem=self.problem,
            placement=previous_placement,
            seed_placement=self.seed_placement,
        )
        next_proxy_cost = self.proxy.predict_cost(
            problem=self.problem,
            placement=next_placement,
            seed_placement=self.seed_placement,
        )
        previous_hpwl = self._hpwl(previous_placement)
        next_hpwl = self._hpwl(next_placement)
        proxy_delta = next_proxy_cost - previous_proxy_cost
        hpwl_delta = next_hpwl - previous_hpwl
        reward = -((self.weights.w_proxy * proxy_delta) + (self.weights.w_hpwl * hpwl_delta))
        if not is_legal:
            reward -= self.weights.legality_penalty
        if len(getattr(next_state, "placed_macros", ())) >= self.movable_macro_count:
            reward += self.weights.completion_bonus
        return float(reward)

    def compute_step_reward(
        self,
        previous_state: Any,
        next_state: Any,
        *,
        is_legal: bool,
    ) -> float:
        raw_reward = self.compute_raw_step_reward(previous_state, next_state, is_legal=is_legal)
        if not self.normalize_rewards:
            return raw_reward
        return float(self.normalizer.normalize(raw_reward))

    def _hpwl(self, placement: dict[str, tuple[float, float]]) -> float:
        if self.hpwl_evaluator is None:
            return float(hpwl_cost(self.problem, placement=placement))
        if hasattr(self.hpwl_evaluator, "evaluate"):
            return float(self.hpwl_evaluator.evaluate(placement))
        return float(self.hpwl_evaluator(placement))


@lru_cache(maxsize=4)
def load_frozen_linear_proxy(
    dataset_path: str | None = None,
    target_name: str = "total_routed_wirelength",
) -> FrozenLinearGraphStatsProxy:
    return FrozenLinearGraphStatsProxy.from_phase2_dataset(dataset_path=dataset_path, target_name=target_name)


def _default_phase2_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "experiments" / "chip" / "phase2_dataset_full_stratified_main.json"
