from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, TypeVar


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class AnnealingResult:
    best_state: object
    best_score: float


class SimulatedAnnealingSolver:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def solve(
        self,
        initial_state: StateT,
        objective_fn: Callable[[StateT], float],
        neighbor_fn: Callable[[StateT, random.Random], StateT],
        iterations: int = 100,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.95,
    ) -> AnnealingResult:
        current_state = initial_state
        current_score = objective_fn(current_state)
        best_state = current_state
        best_score = current_score
        temperature = initial_temperature

        for _ in range(iterations):
            candidate_state = neighbor_fn(current_state, self._rng)
            candidate_score = objective_fn(candidate_state)
            score_delta = candidate_score - current_score

            if score_delta <= 0 or self._accept_worse(score_delta, temperature):
                current_state = candidate_state
                current_score = candidate_score
                if candidate_score < best_score:
                    best_state = candidate_state
                    best_score = candidate_score

            temperature = max(temperature * cooling_rate, 1e-6)

        return AnnealingResult(best_state=best_state, best_score=best_score)

    def _accept_worse(self, score_delta: float, temperature: float) -> bool:
        threshold = math.exp(-score_delta / max(temperature, 1e-6))
        return self._rng.random() < threshold
