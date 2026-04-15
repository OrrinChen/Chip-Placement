from __future__ import annotations

import random
from dataclasses import dataclass

from chip.constraints import LEGALITY_EPSILON
from chip.constraints import build_constraint_registry
from chip.cost import hpwl_cost
from chip.problem import ChipProblem
from engine.constraints import ConstraintViolation
from engine.constraints import ConstraintReport
from engine.graph import ProblemGraph


@dataclass(frozen=True)
class ChipRLState:
    problem: ChipProblem
    placement: dict[str, tuple[float, float]]
    placed_macros: frozenset[str] = frozenset()


def build_macro_order(problem: ChipProblem, strategy: str = "descending_area") -> tuple[str, ...]:
    movable = [macro for macro in problem.macros.values() if not macro.is_fixed]
    if strategy == "descending_area":
        ordered = sorted(movable, key=lambda macro: (-(macro.width * macro.height), macro.name))
        return tuple(macro.name for macro in ordered)
    if strategy == "seed_yx":
        ordered = sorted(movable, key=lambda macro: (macro.y, macro.x, macro.name))
        return tuple(macro.name for macro in ordered)
    raise ValueError(f"Unknown macro ordering strategy: {strategy}")


class ChipRLAdapter:
    def __init__(
        self,
        problem: ChipProblem,
        proxy=None,
        oracle=None,
        reward_model=None,
        hpwl_evaluator=None,
        legality_penalty: float = 100.0,
        min_spacing: float = 0.0,
        grid_shape: tuple[int, int] = (32, 32),
        macro_order: tuple[str, ...] | None = None,
    ) -> None:
        self.problem = problem
        self.proxy = proxy
        self.oracle = oracle
        self.reward_model = reward_model
        self.hpwl_evaluator = hpwl_evaluator
        self.legality_penalty = legality_penalty
        self.grid_shape = grid_shape
        self.min_spacing = min_spacing
        self.constraint_registry = build_constraint_registry(problem, min_spacing=min_spacing)
        self._movable_macros = tuple(name for name, macro in problem.macros.items() if not macro.is_fixed)
        self._macro_order = tuple(macro_order) if macro_order is not None else build_macro_order(problem, "descending_area")
        self._base_edges = self._build_edges(problem)

    @property
    def movable_macros(self) -> tuple[str, ...]:
        return self._movable_macros

    def ordered_macros(self) -> tuple[str, ...]:
        return self._macro_order

    def current_macro_for_step(self, step_index: int) -> str | None:
        ordered = self.ordered_macros()
        return ordered[step_index] if 0 <= step_index < len(ordered) else None

    def initial_state(self) -> ChipRLState:
        return ChipRLState(problem=self.problem, placement=self.problem.placement())

    def observe(self, state: ChipRLState, step_index: int) -> dict[str, object]:
        current_macro = self.current_macro_for_step(step_index)
        action_mask = self.build_action_mask(state, step_index=step_index, grid_shape=self.grid_shape)
        return {
            "step_index": step_index,
            "current_macro": current_macro,
            "current_macro_index": step_index,
            "placed_count": len(state.placed_macros),
            "graph": self._build_graph(state, step_index=step_index),
            "canvas": self.build_canvas(state, step_index=step_index, grid_shape=self.grid_shape, action_mask=action_mask),
            "action_mask": action_mask,
            "placement": dict(state.placement),
            "movable_macros": self.movable_macros,
            "hpwl": self.current_hpwl(state),
        }

    def render(self, state: ChipRLState) -> dict[str, object]:
        return {
            "macro_positions": dict(state.placement),
            "hpwl": self.current_hpwl(state),
            "is_valid": self.constraint_report(state).is_valid,
        }

    def apply_action(self, state: ChipRLState, action: dict[str, object]) -> ChipRLState:
        macro_name = str(action["macro"])
        if macro_name not in self.problem.macros:
            raise KeyError(f"Unknown macro: {macro_name}")
        if self.problem.macros[macro_name].is_fixed:
            return state
        target_x, target_y = action["target"]
        snapped_x, snapped_y, _ = state.problem.snap_movable_macro(macro_name, float(target_x), float(target_y))
        next_placement = dict(state.placement)
        next_placement[macro_name] = (snapped_x, snapped_y)
        return ChipRLState(
            problem=state.problem,
            placement=next_placement,
            placed_macros=state.placed_macros | {macro_name},
        )

    def compute_reward(self, previous_state: ChipRLState, next_state: ChipRLState, reward_source: str) -> float:
        if reward_source == "analytic_hpwl":
            return self.current_hpwl(previous_state) - self.current_hpwl(next_state)
        if reward_source == "proxy":
            return -self._proxy_cost(next_state)
        if reward_source == "proxy_delta":
            return self.proxy_delta(previous_state, next_state, len(previous_state.placed_macros))
        if reward_source == "phase3_proxy_hpwl":
            if self.reward_model is None:
                raise ValueError("Phase 3 reward requested without a reward model")
            return float(
                self.reward_model.compute_step_reward(
                    previous_state,
                    next_state,
                    is_legal=self.constraint_report(next_state).is_valid,
                )
            )
        if reward_source == "oracle":
            return -self._oracle_cost(next_state)
        if reward_source == "proxy_with_legality_penalty":
            reward = -self._proxy_cost(next_state)
            if not self.constraint_report(next_state).is_valid:
                reward -= self.legality_penalty
            return reward
        raise ValueError(f"Unknown reward source: {reward_source}")

    def is_terminal(self, state: ChipRLState, step_index: int, max_steps: int) -> bool:
        del state, max_steps
        return self.current_macro_for_step(step_index) is None

    def current_hpwl(self, state: ChipRLState) -> float:
        if self.hpwl_evaluator is not None:
            if hasattr(self.hpwl_evaluator, "evaluate"):
                return float(self.hpwl_evaluator.evaluate(state.placement))
            return float(self.hpwl_evaluator(state.placement))
        return hpwl_cost(state.problem, placement=state.placement)

    def constraint_report(self, state: ChipRLState) -> ConstraintReport:
        active_names = self._active_macro_names(state)
        return self._constraint_report_for_names(state.placement, active_names)

    def sample_random_action(self, state: ChipRLState, rng: random.Random) -> dict[str, object]:
        current_macro = self.current_macro_for_step(len(state.placed_macros))
        if current_macro is None:
            raise ValueError("Need at least one movable macro to sample actions")
        action_mask = self.build_action_mask(state, step_index=len(state.placed_macros), grid_shape=self.grid_shape)
        valid_indices = [(row_index, col_index) for row_index in range(action_mask.shape[0]) for col_index in range(action_mask.shape[1]) if not bool(action_mask[row_index, col_index])]
        if not valid_indices:
            raise ValueError("Need at least one valid action to sample")
        row_index, col_index = rng.choice(valid_indices)
        target_x, target_y = self.grid_index_to_coordinate((row_index, col_index), self.grid_shape)
        return {"macro": current_macro, "target": (target_x, target_y)}

    def seed_completed_placement(self, state: ChipRLState, step_index: int) -> dict[str, tuple[float, float]]:
        del step_index
        if self.reward_model is not None and hasattr(self.reward_model, "seed_complete"):
            return dict(
                self.reward_model.seed_complete(
                    state.placement,
                    placed_macros=state.placed_macros,
                )
            )
        return dict(state.placement)

    def reset_reward_state(self) -> None:
        if self.reward_model is not None and hasattr(self.reward_model, "reset_episode"):
            self.reward_model.reset_episode()

    def proxy_delta(self, previous_state: ChipRLState, next_state: ChipRLState, step_index: int) -> float:
        previous_cost = self._proxy_cost(
            ChipRLState(
                problem=previous_state.problem,
                placement=self.seed_completed_placement(previous_state, step_index),
                placed_macros=previous_state.placed_macros,
            )
        )
        next_cost = self._proxy_cost(
            ChipRLState(
                problem=next_state.problem,
                placement=self.seed_completed_placement(next_state, step_index + 1),
                placed_macros=next_state.placed_macros,
            )
        )
        return -(next_cost - previous_cost)

    def build_canvas(
        self,
        state: ChipRLState,
        *,
        step_index: int,
        grid_shape: tuple[int, int],
        action_mask=None,
    ):
        import torch

        rows, cols = grid_shape
        occupancy = torch.zeros((rows, cols), dtype=torch.float32)
        fixed_mask = torch.zeros((rows, cols), dtype=torch.float32)
        placed_mask = torch.zeros((rows, cols), dtype=torch.float32)
        if action_mask is None:
            action_mask = self.build_action_mask(state, step_index=step_index, grid_shape=grid_shape)
        legal_mask = (~action_mask).to(dtype=torch.float32)
        y_coords = torch.linspace(0.0, 1.0, steps=rows, dtype=torch.float32).unsqueeze(1).repeat(1, cols)
        x_coords = torch.linspace(0.0, 1.0, steps=cols, dtype=torch.float32).unsqueeze(0).repeat(rows, 1)

        active_names = self._active_macro_names(state)
        for name, macro in state.problem.macros.items():
            if name not in active_names:
                continue
            x, y = state.placement[name]
            row_index, col_index = self.coordinate_to_grid_index((x, y), grid_shape)
            occupancy[row_index, col_index] = 1.0
            if macro.is_fixed:
                fixed_mask[row_index, col_index] = 1.0
            if name in state.placed_macros:
                placed_mask[row_index, col_index] = 1.0

        return torch.stack([occupancy, fixed_mask, placed_mask, legal_mask, x_coords, y_coords], dim=0)

    def build_action_mask(
        self,
        state: ChipRLState,
        *,
        step_index: int,
        grid_shape: tuple[int, int],
    ):
        import torch

        rows, cols = grid_shape
        mask = torch.ones((rows, cols), dtype=torch.bool)
        current_macro = self.current_macro_for_step(step_index)
        if current_macro is None:
            return mask
        for row_index in range(rows):
            for col_index in range(cols):
                target_x, target_y = self.grid_index_to_coordinate((row_index, col_index), grid_shape)
                candidate_state = self.apply_action(state, {"macro": current_macro, "target": (target_x, target_y)})
                mask[row_index, col_index] = not self.constraint_report(candidate_state).is_valid
        return mask

    def grid_index_to_coordinate(
        self,
        index: tuple[int, int],
        grid_shape: tuple[int, int],
    ) -> tuple[float, float]:
        row_index, col_index = index
        rows, cols = grid_shape
        die = self.problem.die_area
        x_step = die.width / max(cols, 1)
        y_step = die.height / max(rows, 1)
        x = die.min_x + ((col_index + 0.5) * x_step)
        y = die.min_y + ((row_index + 0.5) * y_step)
        return (x, y)

    def coordinate_to_grid_index(
        self,
        coordinate: tuple[float, float],
        grid_shape: tuple[int, int],
    ) -> tuple[int, int]:
        x, y = coordinate
        rows, cols = grid_shape
        die = self.problem.die_area
        x_fraction = 0.0 if die.width <= 0 else (x - die.min_x) / die.width
        y_fraction = 0.0 if die.height <= 0 else (y - die.min_y) / die.height
        col_index = min(max(int(x_fraction * cols), 0), cols - 1)
        row_index = min(max(int(y_fraction * rows), 0), rows - 1)
        return (row_index, col_index)

    def _build_graph(self, state: ChipRLState, *, step_index: int) -> ProblemGraph:
        current_macro = self.current_macro_for_step(step_index)
        nodes = {}
        for name, macro in state.problem.macros.items():
            x, y = state.placement[name]
            nodes[name] = {
                "kind": macro.kind,
                "width": macro.width,
                "height": macro.height,
                "x": x,
                "y": y,
                "orientation": macro.orientation,
                "is_fixed": macro.is_fixed,
                "placed_flag": 1.0 if name in state.placed_macros else 0.0,
                "current_macro_flag": 1.0 if name == current_macro else 0.0,
            }
        return ProblemGraph(
            nodes=nodes,
            edges=list(self._base_edges),
            global_features={
                "bounds": (
                    state.problem.die_area.min_x,
                    state.problem.die_area.max_x,
                    state.problem.die_area.min_y,
                    state.problem.die_area.max_y,
                ),
                "fixed_nodes": {name for name, macro in state.problem.macros.items() if macro.is_fixed},
            },
        )

    def _proxy_cost(self, state: ChipRLState) -> float:
        if self.proxy is None:
            raise ValueError("Proxy reward requested without a proxy model")
        return float(self.proxy.predict(self.observe(state, step_index=0)))

    def _oracle_cost(self, state: ChipRLState) -> float:
        if self.oracle is None:
            raise ValueError("Oracle reward requested without an evaluator")
        return float(self.oracle.evaluate(state))

    def _active_macro_names(self, state: ChipRLState) -> set[str]:
        return {
            name
            for name, macro in self.problem.macros.items()
            if macro.is_fixed or name in state.placed_macros
        }

    def _constraint_report_for_names(
        self,
        placement: dict[str, tuple[float, float]],
        active_names: set[str],
    ) -> ConstraintReport:
        violations: list[ConstraintViolation] = []
        die = self.problem.die_area
        rectangles = []
        active_macros = [self.problem.macros[name] for name in active_names]

        for macro in active_macros:
            x, y = placement[macro.name]
            overflow = max(die.min_x - x, 0.0) + max(die.min_y - y, 0.0)
            overflow += max((x + macro.width) - die.max_x, 0.0)
            overflow += max((y + macro.height) - die.max_y, 0.0)
            if overflow > LEGALITY_EPSILON:
                violations.append(
                    ConstraintViolation(
                        name="boundary",
                        magnitude=float(overflow),
                        penalty_weight=5.0,
                        details={"macro": macro.name},
                    )
                )
            if macro.width > 0.0 and macro.height > 0.0:
                rectangles.append((x, x + macro.width, y, y + macro.height, macro.name))

        rectangles.sort(key=lambda rect: rect[0])
        active_rectangles: list[tuple[float, float, float, float, str]] = []
        for x1, x2, y1, y2, name in rectangles:
            active_rectangles = [rect for rect in active_rectangles if rect[1] > x1]
            for other_x1, other_x2, other_y1, other_y2, other_name in active_rectangles:
                overlap_x = min(x2, other_x2) - max(x1, other_x1)
                overlap_y = min(y2, other_y2) - max(y1, other_y1)
                if overlap_x > LEGALITY_EPSILON and overlap_y > LEGALITY_EPSILON:
                    violations.append(
                        ConstraintViolation(
                            name="non_overlap",
                            magnitude=float(overlap_x * overlap_y),
                            penalty_weight=10.0,
                            details={"first": other_name, "second": name},
                        )
                    )
            active_rectangles.append((x1, x2, y1, y2, name))

        if self.min_spacing > 0.0:
            for index, first in enumerate(active_macros):
                x1, y1 = placement[first.name]
                for second in active_macros[index + 1 :]:
                    x2, y2 = placement[second.name]
                    gap_x = max(x2 - (x1 + first.width), x1 - (x2 + second.width), 0.0)
                    gap_y = max(y2 - (y1 + first.height), y1 - (y2 + second.height), 0.0)
                    if gap_x == 0.0 and gap_y == 0.0:
                        continue
                    if gap_x < self.min_spacing and gap_y < self.min_spacing:
                        violations.append(
                            ConstraintViolation(
                                name="min_spacing",
                                magnitude=float(self.min_spacing - min(gap_x, gap_y)),
                                penalty_weight=3.0,
                                details={"first": first.name, "second": second.name},
                            )
                        )
        return ConstraintReport(violations=violations)

    @staticmethod
    def _build_edges(problem: ChipProblem) -> tuple[tuple[str, str, dict[str, float]], ...]:
        edge_weights: dict[tuple[str, str], float] = {}
        for net in problem.nets:
            known_pins = [pin for pin in net.pins if pin in problem.macros]
            if len(known_pins) < 2:
                continue
            anchor = known_pins[0]
            for target in known_pins[1:]:
                key = (anchor, target)
                edge_weights[key] = edge_weights.get(key, 0.0) + 1.0
        return tuple((source, target, {"weight": weight}) for (source, target), weight in edge_weights.items())
