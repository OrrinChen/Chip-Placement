from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from collections import deque
from pathlib import Path
from statistics import mean

import torch

from chip.phase3_reward import RewardWeights
from chip.experiments.phase3_baselines import run_greedy_episode
from chip.experiments.phase3_macroplacement_bringup import build_ariane_formulation_config
from chip.experiments.phase3_macroplacement_bringup import build_ariane_problem
from chip.experiments.phase3_macroplacement_bringup import build_ariane_reward_model
from chip.experiments.phase3_ppo import FixedRolloutBatch
from chip.experiments.phase3_ppo import collect_policy_rollout
from chip.experiments.phase3_ppo import collect_rollout_for_action_trace
from chip.experiments.phase3_ppo import initialize_policy_for_problem
from chip.experiments.phase3_ppo import load_policy_warm_start
from chip.experiments.phase3_ppo import overfit_fixed_rollout
from chip.experiments.phase3_ppo import prefit_value_head_from_rollout
from chip.experiments.phase3_ppo import prefit_value_head_from_rollouts
from chip.experiments.phase3_ppo import run_single_ppo_update
from chip.experiments.phase3_ppo import summarize_rollout_return_signal
from chip.experiments.phase3_ppo import train_ppo_policy
from chip.rl_adapter import ChipRLAdapter
from engine.rl_env import PlacementEnv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _reverse_curriculum_status_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_status.json")


def _write_reverse_curriculum_status(
    *,
    output_path: Path,
    status: str,
    stage: str,
    **extra: object,
) -> None:
    payload = {
        "status": status,
        "stage": stage,
        **extra,
    }
    _reverse_curriculum_status_path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def resolve_ariane_warm_start_checkpoint(artifact_dir: Path) -> Path:
    local_checkpoint = artifact_dir / "ariane_greedy_bc.pt"
    if local_checkpoint.exists():
        return local_checkpoint
    fallback = _repo_root() / "experiments" / "chip" / "phase3_macroplacement_warmstart_pilot_v2" / "ariane_greedy_bc.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find Ariane warm-start checkpoint in {local_checkpoint} or {fallback}")


def build_ariane_checkpoint_training_config() -> dict[str, float]:
    return build_ariane_warm_kl_anchor_training_config()


def build_ariane_scratch_training_config() -> dict[str, float]:
    return {
        "init_seed": 0,
        "learning_rate": 1e-3,
        "value_coef": 0.0,
        "entropy_coef": 1.0,
    }


def build_ariane_warm_entropy_training_config() -> dict[str, float]:
    return {
        "init_seed": 0,
        "learning_rate": 1e-4,
        "value_coef": 0.0,
        "entropy_coef": 5.0,
    }


def build_ariane_warm_kl_anchor_training_config() -> dict[str, float]:
    config = build_ariane_warm_entropy_training_config()
    config["kl_anchor_beta"] = 5.0
    return config


def build_ariane_warm_value_prefit_training_config() -> dict[str, float]:
    return {
        "init_seed": 0,
        "learning_rate": 1e-4,
        "value_coef": 0.0,
        "entropy_coef": 1.0,
        "prefit_learning_rate": 1e-3,
        "prefit_epochs": 50,
    }


def _resolve_ariane_grid_shape(grid_shape: tuple[int, int] | None) -> tuple[int, int]:
    return tuple(grid_shape or build_ariane_formulation_config()["grid_shape"])


def _normalize_trace_target(target: tuple[float, float] | list[float]) -> list[float]:
    x, y = target
    return [float(x), float(y)]


def _clone_reference_policy(policy):
    reference_policy = copy.deepcopy(policy)
    if hasattr(reference_policy, "eval"):
        reference_policy.eval()
    return reference_policy


def _training_kl_anchor_beta(training_config: dict[str, float]) -> float:
    return float(training_config.get("kl_anchor_beta", 0.0))


def _reference_policy_for_training(policy, training_config: dict[str, float]):
    if _training_kl_anchor_beta(training_config) <= 0.0:
        return None
    return _clone_reference_policy(policy)


def summarize_termination_audit(
    *,
    ordered_macros: tuple[str, ...],
    trace: list[dict[str, object]],
    termination_reason: str,
    final_hpwl: float,
    episode_return: float,
    is_legal: bool,
) -> dict[str, object]:
    placed_count = len(trace)
    remaining_count = max(0, len(ordered_macros) - placed_count)
    failing_macro = ordered_macros[placed_count] if placed_count < len(ordered_macros) else None
    last_step = None
    if trace:
        last_row = trace[-1]
        last_step = {
            "step_index": int(last_row["step_index"]),
            "macro": str(last_row["macro"]),
            "target": _normalize_trace_target(last_row["target"]),
            "legal_action_count": int(last_row["legal_action_count"]),
        }
    return {
        "trace": [
            {
                "step_index": int(row["step_index"]),
                "macro": str(row["macro"]),
                "target": _normalize_trace_target(row["target"]),
                "legal_action_count": int(row["legal_action_count"]),
            }
            for row in trace
        ],
        "steps": placed_count,
        "placed_macro_count": placed_count,
        "remaining_macro_count": remaining_count,
        "termination_reason": str(termination_reason),
        "completed_episode": termination_reason == "completed",
        "failing_macro": failing_macro,
        "final_hpwl": float(final_hpwl),
        "episode_return": float(episode_return),
        "is_legal": bool(is_legal),
        "last_step": last_step,
    }


def find_first_trace_divergence(
    *,
    reference_trace: list[dict[str, object]],
    candidate_trace: list[dict[str, object]],
) -> dict[str, object] | None:
    limit = min(len(reference_trace), len(candidate_trace))
    for index in range(limit):
        reference = reference_trace[index]
        candidate = candidate_trace[index]
        reference_payload = {
            "step_index": int(reference["step_index"]),
            "macro": str(reference["macro"]),
            "target": _normalize_trace_target(reference["target"]),
        }
        candidate_payload = {
            "step_index": int(candidate["step_index"]),
            "macro": str(candidate["macro"]),
            "target": _normalize_trace_target(candidate["target"]),
        }
        if reference_payload != candidate_payload:
            return {
                "step_index": index,
                "reference": reference_payload,
                "candidate": candidate_payload,
            }
    if len(reference_trace) == len(candidate_trace):
        return None
    shorter = "reference" if len(reference_trace) < len(candidate_trace) else "candidate"
    extra = candidate_trace[limit] if shorter == "reference" else reference_trace[limit]
    return {
        "step_index": limit,
        "reference": None if shorter == "reference" else {
            "step_index": int(extra["step_index"]),
            "macro": str(extra["macro"]),
            "target": _normalize_trace_target(extra["target"]),
        },
        "candidate": None if shorter == "candidate" else {
            "step_index": int(extra["step_index"]),
            "macro": str(extra["macro"]),
            "target": _normalize_trace_target(extra["target"]),
        },
    }


def _build_step_histogram(history: list[dict[str, object]]) -> dict[str, int]:
    counter = Counter(int(item.get("steps", 0)) for item in history)
    return {str(step): int(counter[step]) for step in sorted(counter)}


def _rollout_json_summary(rollout: dict[str, object]) -> dict[str, object]:
    return {
        "steps": int(rollout["steps"]),
        "episode_return": float(rollout["episode_return"]),
        "final_hpwl": float(rollout["final_hpwl"]),
        "invalid_action_count": int(rollout["invalid_action_count"]),
        "termination_reason": str(rollout["termination_reason"]),
        "completed_episode": bool(rollout["completed_episode"]),
        "is_legal": bool(rollout["is_legal"]),
    }


def _load_ariane_reference_levels(artifact_dir: Path) -> tuple[float | None, float | None, list[str]]:
    candidate_roots = [artifact_dir, artifact_dir.parent, _repo_root() / "experiments" / "chip"]
    random_mean_hpwl = None
    greedy_final_hpwl = None
    warnings: list[str] = []
    for root in candidate_roots:
        random_artifact_path = root / "phase3_masked_random_ariane.json"
        if random_mean_hpwl is None and random_artifact_path.exists():
            random_mean_hpwl = float(json.loads(random_artifact_path.read_text(encoding="utf-8-sig"))["summary"]["final_hpwl"]["mean"])
        greedy_artifact_path = root / "phase3_greedy_ariane.json"
        if greedy_final_hpwl is None and greedy_artifact_path.exists():
            greedy_final_hpwl = float(json.loads(greedy_artifact_path.read_text(encoding="utf-8-sig"))["episode"]["final_hpwl"])
    if random_mean_hpwl is None:
        warnings.append("missing_masked_random_reference")
    if greedy_final_hpwl is None:
        warnings.append("missing_greedy_reference")
    return random_mean_hpwl, greedy_final_hpwl, warnings


def _replay_rollout_state(
    *,
    adapter: ChipRLAdapter,
    rollout: dict[str, object],
    grid_shape: tuple[int, int],
):
    if "transitions" not in rollout:
        return None
    state = adapter.initial_state()
    for transition in rollout["transitions"]:
        step_index = len(state.placed_macros)
        macro_name = adapter.current_macro_for_step(step_index)
        if macro_name is None:
            break
        action_index = int(transition.action_index)
        row_index = action_index // grid_shape[1]
        col_index = action_index % grid_shape[1]
        target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
        state = adapter.apply_action(state, {"macro": macro_name, "target": target})
    return state


def _largest_connected_component(mask: list[list[bool]]) -> dict[str, object]:
    rows = len(mask)
    cols = len(mask[0]) if rows else 0
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    best_cells = 0
    best_bbox = None
    component_count = 0
    total_cells = 0
    for row_index in range(rows):
        for col_index in range(cols):
            if not mask[row_index][col_index]:
                total_cells += 1
            if mask[row_index][col_index] or visited[row_index][col_index]:
                continue
            component_count += 1
            queue = deque([(row_index, col_index)])
            visited[row_index][col_index] = True
            cells = 0
            min_row = max_row = row_index
            min_col = max_col = col_index
            while queue:
                current_row, current_col = queue.popleft()
                cells += 1
                min_row = min(min_row, current_row)
                max_row = max(max_row, current_row)
                min_col = min(min_col, current_col)
                max_col = max(max_col, current_col)
                for delta_row, delta_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                        continue
                    if mask[next_row][next_col] or visited[next_row][next_col]:
                        continue
                    visited[next_row][next_col] = True
                    queue.append((next_row, next_col))
            if cells > best_cells:
                best_cells = cells
                best_bbox = {
                    "min_row": min_row,
                    "max_row": max_row,
                    "min_col": min_col,
                    "max_col": max_col,
                }
    return {
        "component_count": component_count,
        "total_free_cells": total_cells,
        "largest_component_cells": best_cells,
        "largest_component_bbox": best_bbox,
    }


def _summarize_remaining_macro_geometry(
    *,
    adapter: ChipRLAdapter,
    state,
    grid_shape: tuple[int, int],
) -> dict[str, object]:
    ordered = adapter.ordered_macros()
    placed_count = len(state.placed_macros)
    remaining_names = ordered[placed_count:]
    remaining_area = 0.0
    bbox = None
    if remaining_names:
        min_x = min(state.placement[name][0] for name in remaining_names)
        min_y = min(state.placement[name][1] for name in remaining_names)
        max_x = max(state.placement[name][0] + adapter.problem.macros[name].width for name in remaining_names)
        max_y = max(state.placement[name][1] + adapter.problem.macros[name].height for name in remaining_names)
        remaining_area = float(
            sum(adapter.problem.macros[name].width * adapter.problem.macros[name].height for name in remaining_names)
        )
        bbox = {
            "min_x": float(min_x),
            "min_y": float(min_y),
            "max_x": float(max_x),
            "max_y": float(max_y),
            "width": float(max_x - min_x),
            "height": float(max_y - min_y),
            "area": float((max_x - min_x) * (max_y - min_y)),
        }
    active_names = adapter._active_macro_names(state)
    rectangles = []
    for name in active_names:
        macro = adapter.problem.macros[name]
        x, y = state.placement[name]
        rectangles.append((x, x + macro.width, y, y + macro.height))
    blocked = [[False for _ in range(grid_shape[1])] for _ in range(grid_shape[0])]
    for row_index in range(grid_shape[0]):
        for col_index in range(grid_shape[1]):
            x, y = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
            for x1, x2, y1, y2 in rectangles:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    blocked[row_index][col_index] = True
                    break
    free_components = _largest_connected_component(blocked)
    die = adapter.problem.die_area
    cell_area = (die.width / max(grid_shape[1], 1)) * (die.height / max(grid_shape[0], 1))
    largest_component_area = float(free_components["largest_component_cells"] * cell_area)
    total_free_area = float(free_components["total_free_cells"] * cell_area)
    largest_vs_remaining = None
    if remaining_area > 0:
        largest_vs_remaining = largest_component_area / remaining_area
    return {
        "macro_total_count": len(ordered),
        "placed_macro_count": placed_count,
        "remaining_macro_count": len(remaining_names),
        "remaining_macro_names_head": list(remaining_names[:10]),
        "remaining_macro_total_area": float(remaining_area),
        "remaining_macro_bbox": bbox,
        "free_space": {
            **free_components,
            "largest_component_area": largest_component_area,
            "total_free_area": total_free_area,
            "largest_component_vs_remaining_area_ratio": largest_vs_remaining,
        },
    }


def _summarize_rollout_geometry_from_rollout(
    *,
    adapter: ChipRLAdapter,
    rollout: dict[str, object],
    grid_shape: tuple[int, int],
) -> dict[str, object]:
    state = _replay_rollout_state(adapter=adapter, rollout=rollout, grid_shape=grid_shape)
    if state is None:
        total_macros = len(adapter.ordered_macros())
        placed_count = int(rollout.get("steps", 0))
        return {
            "macro_total_count": total_macros,
            "placed_macro_count": placed_count,
            "remaining_macro_count": max(total_macros - placed_count, 0),
            "remaining_macro_names_head": list(adapter.ordered_macros()[placed_count : placed_count + 10]),
            "remaining_macro_total_area": None,
            "remaining_macro_bbox": None,
            "free_space": None,
        }
    return _summarize_remaining_macro_geometry(adapter=adapter, state=state, grid_shape=grid_shape)


def _iter_rollout_action_indices(rollout: dict[str, object], *, limit: int | None = None) -> list[int]:
    action_indices: list[int] = []
    transitions = rollout.get("transitions")
    if transitions:
        action_indices = [int(transition.action_index) for transition in transitions]
    else:
        raw_indices = rollout.get("action_indices", [])
        action_indices = [int(action_index) for action_index in raw_indices]
    if limit is not None:
        return action_indices[: max(0, limit)]
    return action_indices


def _summarize_export_rollout_trace(
    *,
    adapter: ChipRLAdapter,
    rollout: dict[str, object],
    grid_shape: tuple[int, int],
) -> dict[str, object]:
    action_indices = _iter_rollout_action_indices(rollout)
    reward_trace: list[float] = []
    value_trace: list[float] = []
    hpwl_trace: list[float] = []
    transitions = rollout.get("transitions") or []
    for transition in transitions:
        reward_trace.append(float(getattr(transition, "reward", 0.0)))
        value_trace.append(float(getattr(transition, "value", 0.0)))
    try:
        state = adapter.initial_state()
        for action_index in action_indices:
            step_index = len(state.placed_macros)
            macro_name = adapter.current_macro_for_step(step_index)
            if macro_name is None:
                break
            row_index = int(action_index) // grid_shape[1]
            col_index = int(action_index) % grid_shape[1]
            target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
            state = adapter.apply_action(state, {"macro": macro_name, "target": target})
            hpwl_trace.append(float(adapter.current_hpwl(state)))
    except Exception:
        hpwl_trace = []
    return {
        "action_indices": [int(action_index) for action_index in action_indices],
        "reward_trace": reward_trace,
        "value_trace": value_trace,
        "hpwl_trace": hpwl_trace,
        "transition_count": int(len(action_indices)),
    }


def _replay_prefix_state_from_rollout(
    *,
    problem,
    reward_model,
    grid_shape: tuple[int, int],
    rollout: dict[str, object],
    prefix_steps: int,
):
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=grid_shape,
    )
    state = adapter.initial_state()
    action_indices = _iter_rollout_action_indices(rollout, limit=prefix_steps)
    for action_index in action_indices:
        step_index = len(state.placed_macros)
        macro_name = adapter.current_macro_for_step(step_index)
        if macro_name is None:
            break
        row_index = action_index // grid_shape[1]
        col_index = action_index % grid_shape[1]
        target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
        state = adapter.apply_action(state, {"macro": macro_name, "target": target})
    return state


def _run_episode_from_prefix_state(
    *,
    problem,
    reward_model,
    grid_shape: tuple[int, int],
    prefix_state,
    reward_source: str,
    strategy: str,
    seed: int = 0,
) -> dict[str, object]:
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=grid_shape,
    )
    env = PlacementEnv(adapter=adapter, max_steps=len(adapter.ordered_macros()), reward_source=reward_source)
    rng = random.Random(seed)
    observation, _ = env.reset(
        seed=seed,
        options={
            "initial_state": prefix_state,
            "initial_step_index": len(getattr(prefix_state, "placed_macros", ())),
        },
    )
    episode_return = 0.0
    termination_reason = "completed"

    while True:
        current_state = env.state
        step_index = len(current_state.placed_macros)
        current_macro = adapter.current_macro_for_step(step_index)
        if current_macro is None:
            termination_reason = "completed"
            break
        action_mask = adapter.build_action_mask(current_state, step_index=step_index, grid_shape=grid_shape)
        if bool(action_mask.all()):
            termination_reason = "no_valid_actions"
            break

        if strategy == "random":
            action = adapter.sample_random_action(current_state, rng)
        elif strategy == "greedy":
            best_action = None
            best_hpwl = None
            for row_index in range(grid_shape[0]):
                for col_index in range(grid_shape[1]):
                    if bool(action_mask[row_index, col_index]):
                        continue
                    target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
                    candidate_action = {"macro": current_macro, "target": target}
                    candidate_state = adapter.apply_action(current_state, candidate_action)
                    candidate_hpwl = adapter.current_hpwl(candidate_state)
                    if best_hpwl is None or candidate_hpwl < best_hpwl:
                        best_hpwl = candidate_hpwl
                        best_action = candidate_action
            if best_action is None:
                termination_reason = "no_valid_actions"
                break
            action = best_action
        else:
            raise ValueError(f"Unsupported suffix strategy: {strategy}")

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            termination_reason = "completed" if terminated else "truncated"
            break

    final_state = env.state
    return {
        "steps": int(len(getattr(final_state, "placed_macros", ()))),
        "episode_return": float(episode_return),
        "final_hpwl": float(adapter.current_hpwl(final_state)),
        "termination_reason": termination_reason,
        "completed_episode": termination_reason == "completed",
        "is_legal": bool(adapter.constraint_report(final_state).is_valid),
    }


def _evaluate_prefix_candidate_salvage(
    *,
    problem,
    reward_model,
    grid_shape: tuple[int, int],
    prefix_state,
    reward_source: str = "phase3_proxy_hpwl",
) -> dict[str, object]:
    return {
        "greedy": _run_episode_from_prefix_state(
            problem=problem,
            reward_model=reward_model,
            grid_shape=grid_shape,
            prefix_state=prefix_state,
            reward_source=reward_source,
            strategy="greedy",
            seed=0,
        ),
        "random": _run_episode_from_prefix_state(
            problem=problem,
            reward_model=reward_model,
            grid_shape=grid_shape,
            prefix_state=prefix_state,
            reward_source=reward_source,
            strategy="random",
            seed=0,
        ),
    }


def _instantiate_ariane_warm_policy(
    *,
    problem,
    grid_shape: tuple[int, int],
    hidden_dim: int,
    checkpoint: Path,
    fallback_policy=None,
):
    try:
        policy = initialize_policy_for_problem(problem=problem, grid_shape=grid_shape, hidden_dim=hidden_dim)
    except Exception:
        if fallback_policy is None:
            raise
        policy = fallback_policy
    load_policy_warm_start(policy, checkpoint)
    return policy


def _geometry_ratio_from_summary(geometry: dict[str, object] | None) -> float | None:
    if not geometry:
        return None
    free_space = geometry.get("free_space")
    if not isinstance(free_space, dict):
        return None
    ratio = free_space.get("largest_component_vs_remaining_area_ratio")
    if ratio is None:
        return None
    return float(ratio)


def _normalize_hpwl_threshold(threshold: float | None) -> float | None:
    if threshold is None:
        return None
    value = float(threshold)
    if 0.0 < abs(value) < 10_000.0:
        return float(value * 1_000_000.0)
    return value


def _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep: tuple[float, ...] | None) -> tuple[float, ...]:
    if hpwl_threshold_sweep is None:
        return ()
    return tuple(float(_normalize_hpwl_threshold(value)) for value in hpwl_threshold_sweep)


def _evaluate_prefix_acceptance(
    *,
    rollout: dict[str, object],
    geometry_summary: dict[str, object] | None,
    hpwl_threshold: float,
    remaining_macro_range: tuple[int, int],
    geometry_packing_factor: float,
) -> dict[str, object]:
    remaining_macro_count = (
        int(geometry_summary["remaining_macro_count"])
        if geometry_summary and geometry_summary.get("remaining_macro_count") is not None
        else None
    )
    geometry_ratio = _geometry_ratio_from_summary(geometry_summary)
    hpwl_ok = float(rollout["final_hpwl"]) <= float(_normalize_hpwl_threshold(hpwl_threshold))
    remaining_ok = (
        remaining_macro_count is not None
        and int(remaining_macro_range[0]) <= remaining_macro_count <= int(remaining_macro_range[1])
    )
    geometry_ok = geometry_ratio is not None and geometry_ratio >= float(geometry_packing_factor)
    accepted = (
        not bool(rollout["completed_episode"])
        and hpwl_ok
        and remaining_ok
        and geometry_ok
    )
    return {
        "hpwl_ok": bool(hpwl_ok),
        "remaining_macro_ok": bool(remaining_ok),
        "geometry_feasible": bool(geometry_ok),
        "accepted_for_reverse_curriculum": bool(accepted),
        "remaining_macro_count": remaining_macro_count,
        "geometry_packing_ratio": geometry_ratio,
    }


def _summarize_prefix_acceptance(
    *,
    rollouts: list[dict[str, object]],
    min_accepted_prefixes: int,
    hpwl_threshold_sweep: tuple[float, ...] | None = None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None = None,
    geometry_packing_factor: float = 1.0,
) -> dict[str, object]:
    total_rollouts = len(rollouts)
    accepted_prefixes = sum(1 for row in rollouts if bool(row.get("accepted_for_reverse_curriculum")))
    hpwl_eligible = sum(1 for row in rollouts if bool(row.get("hpwl_ok")))
    remaining_eligible = sum(1 for row in rollouts if bool(row.get("remaining_macro_ok")))
    geometry_eligible = sum(1 for row in rollouts if bool(row.get("geometry_feasible")))
    unique_source_rollouts = len(
        {
            str(row.get("source_rollout_id"))
            for row in rollouts
            if bool(row.get("accepted_for_reverse_curriculum")) and row.get("source_rollout_id") is not None
        }
    )
    meets_min_prefix_count = accepted_prefixes >= int(min_accepted_prefixes)
    if meets_min_prefix_count:
        status = "pass"
    elif accepted_prefixes > 0:
        status = "partial"
    else:
        status = "fail"
    return {
        "total_rollouts": int(total_rollouts),
        "accepted_prefixes": int(accepted_prefixes),
        "hpwl_eligible": int(hpwl_eligible),
        "remaining_eligible": int(remaining_eligible),
        "geometry_eligible": int(geometry_eligible),
        "unique_source_rollouts": int(unique_source_rollouts),
        "meets_min_prefix_count": bool(meets_min_prefix_count),
        "status": status,
        "threshold_sweep": _build_prefix_acceptance_threshold_sweep(
            rollouts=rollouts,
            hpwl_threshold_sweep=hpwl_threshold_sweep,
            remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
            geometry_packing_factor=geometry_packing_factor,
        ),
    }


def _build_prefix_acceptance_threshold_sweep(
    *,
    rollouts: list[dict[str, object]],
    hpwl_threshold_sweep: tuple[float, ...] | None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None,
    geometry_packing_factor: float,
) -> list[dict[str, object]]:
    thresholds = tuple(float(value) for value in (hpwl_threshold_sweep or ()))
    raw_thresholds = _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep)
    remaining_ranges = tuple((int(bounds[0]), int(bounds[1])) for bounds in (remaining_macro_ranges_sweep or ()))
    if not thresholds or not remaining_ranges:
        return []
    cells: list[dict[str, object]] = []
    for hpwl_threshold, hpwl_threshold_raw in zip(thresholds, raw_thresholds):
        for remaining_range in remaining_ranges:
            accepted = [
                row
                for row in rollouts
                if (
                    not bool(row.get("completed_episode"))
                    and float(row.get("final_hpwl", float("inf"))) <= hpwl_threshold_raw
                    and row.get("remaining_macro_count") is not None
                    and remaining_range[0] <= int(row["remaining_macro_count"]) <= remaining_range[1]
                    and row.get("geometry_packing_ratio") is not None
                    and float(row["geometry_packing_ratio"]) >= float(geometry_packing_factor)
                )
            ]
            cells.append(
                {
                    "hpwl_threshold": float(hpwl_threshold),
                    "remaining_macro_range": [int(remaining_range[0]), int(remaining_range[1])],
                    "accepted_prefixes": int(len(accepted)),
                    "unique_source_rollouts": int(
                        len({str(row.get("source_rollout_id")) for row in accepted if row.get("source_rollout_id") is not None})
                    ),
                }
            )
    return cells


def _interpolated_quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_scalar_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    cast = [float(value) for value in values]
    return {
        "count": int(len(cast)),
        "min": float(min(cast)),
        "p10": _interpolated_quantile(cast, 0.10),
        "median": _interpolated_quantile(cast, 0.50),
        "p90": _interpolated_quantile(cast, 0.90),
        "max": float(max(cast)),
        "mean": float(mean(cast)),
    }


def _build_hpwl_threshold_hit_rates(
    *,
    rollouts: list[dict[str, object]],
    hpwl_threshold_sweep: tuple[float, ...] | None,
) -> list[dict[str, object]]:
    if hpwl_threshold_sweep is None:
        return []
    total = max(1, len(rollouts))
    rows: list[dict[str, object]] = []
    raw_thresholds = _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep)
    for threshold, raw_threshold in zip(hpwl_threshold_sweep, raw_thresholds):
        hit_count = sum(1 for row in rollouts if float(row.get("final_hpwl", float("inf"))) <= raw_threshold)
        rows.append(
            {
                "hpwl_threshold": float(threshold),
                "count": int(hit_count),
                "hit_rate": float(hit_count / total),
            }
        )
    return rows


def _build_export_rollout_row(
    *,
    adapter: ChipRLAdapter,
    rollout: dict[str, object],
    grid_shape: tuple[int, int],
    source_rollout_id: str,
    hpwl_threshold: float,
    remaining_macro_range: tuple[int, int],
    geometry_packing_factor: float,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        geometry_summary = _summarize_rollout_geometry_from_rollout(
            adapter=adapter,
            rollout=rollout,
            grid_shape=grid_shape,
        )
    except Exception:
        geometry_summary = None
    acceptance = _evaluate_prefix_acceptance(
        rollout=rollout,
        geometry_summary=geometry_summary,
        hpwl_threshold=hpwl_threshold,
        remaining_macro_range=remaining_macro_range,
        geometry_packing_factor=geometry_packing_factor,
    )
    return {
        "source_rollout_id": source_rollout_id,
        "steps": int(rollout["steps"]),
        "final_hpwl": float(rollout["final_hpwl"]),
        "completed_episode": bool(rollout["completed_episode"]),
        "termination_reason": str(rollout["termination_reason"]),
        "action_indices": _iter_rollout_action_indices(rollout),
        "rollout_trace": _summarize_export_rollout_trace(
            adapter=adapter,
            rollout=rollout,
            grid_shape=grid_shape,
        ),
        **acceptance,
        "geometry_summary": geometry_summary,
        **({} if extra_fields is None else extra_fields),
    }


def _summarize_rollout_probe_distribution(
    *,
    rollouts: list[dict[str, object]],
    hpwl_threshold_sweep: tuple[float, ...] | None,
) -> dict[str, object]:
    return {
        "final_hpwl": _summarize_scalar_distribution([float(row["final_hpwl"]) for row in rollouts]),
        "steps": _summarize_scalar_distribution([float(row["steps"]) for row in rollouts]),
        "threshold_hit_rates": _build_hpwl_threshold_hit_rates(
            rollouts=rollouts,
            hpwl_threshold_sweep=hpwl_threshold_sweep,
        ),
    }


def _summarize_rollout_probe_cohort(
    *,
    rollouts: list[dict[str, object]],
    hpwl_threshold_sweep: tuple[float, ...] | None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None,
    geometry_packing_factor: float,
    min_accepted_prefixes: int,
) -> dict[str, object]:
    completed_count = sum(1 for row in rollouts if bool(row.get("completed_episode")))
    return {
        "rollout_count": int(len(rollouts)),
        "completed_count": int(completed_count),
        "completion_rate": float(completed_count / max(1, len(rollouts))),
        "distribution": _summarize_rollout_probe_distribution(
            rollouts=rollouts,
            hpwl_threshold_sweep=hpwl_threshold_sweep,
        ),
        "acceptance_summary": _summarize_prefix_acceptance(
            rollouts=rollouts,
            min_accepted_prefixes=min_accepted_prefixes,
            hpwl_threshold_sweep=hpwl_threshold_sweep,
            remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
            geometry_packing_factor=geometry_packing_factor,
        ),
    }


def _calibrate_rollout_rows_for_salvage(
    *,
    problem,
    reward_model,
    metadata: dict[str, object],
    grid_shape: tuple[int, int],
    rollouts: list[dict[str, object]],
    hpwl_threshold: float,
    tail_steps: int,
) -> dict[str, object]:
    hpwl_threshold_raw = float(_normalize_hpwl_threshold(hpwl_threshold))
    prefixes: list[dict[str, object]] = []
    greedy_completable = 0
    random_completable = 0
    for row in rollouts:
        if bool(row["completed_episode"]) or int(row["steps"]) <= int(tail_steps):
            continue
        if float(row["final_hpwl"]) > hpwl_threshold_raw:
            continue
        prefix_steps = int(row["steps"]) - int(tail_steps)
        prefix_state = _replay_prefix_state_from_rollout(
            problem=problem,
            reward_model=reward_model,
            grid_shape=grid_shape,
            rollout=row,
            prefix_steps=prefix_steps,
        )
        salvage = _evaluate_prefix_candidate_salvage(
            problem=problem,
            reward_model=reward_model,
            grid_shape=grid_shape,
            prefix_state=prefix_state,
            reward_source="phase3_proxy_hpwl",
        )
        if bool(salvage["greedy"]["completed_episode"]):
            greedy_completable += 1
        if bool(salvage["random"]["completed_episode"]):
            random_completable += 1
        prefixes.append(
            {
                "name": row.get("name", row["source_rollout_id"]),
                "source_rollout_id": row["source_rollout_id"],
                "steps": int(row["steps"]),
                "final_hpwl": float(row["final_hpwl"]),
                "prefix_steps": int(prefix_steps),
                "remaining_steps": int(metadata["ordered_macro_count"]) - int(prefix_steps),
                "salvage": salvage,
            }
        )
    recommended_mode = "greedy_or_random" if (greedy_completable > 0 or random_completable > 0) else "none"
    return {
        "tail_steps": int(tail_steps),
        "prefixes": prefixes,
        "summary": {
            "total_candidates": int(len(prefixes)),
            "greedy_completable": int(greedy_completable),
            "random_completable": int(random_completable),
            "recommended_salvage_mode": recommended_mode,
        },
    }


def _load_external_prefix_candidates(
    *,
    prefix_source_path: Path,
    hpwl_threshold: float | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(prefix_source_path.read_text(encoding="utf-8-sig"))
    rollouts = [dict(row) for row in payload.get("rollouts", [])]
    hpwl_threshold_raw = _normalize_hpwl_threshold(hpwl_threshold)
    if hpwl_threshold is not None:
        rollouts = [row for row in rollouts if float(row.get("final_hpwl", float("inf"))) <= float(hpwl_threshold_raw)]
    rollouts.sort(key=lambda row: float(row.get("final_hpwl", float("inf"))))
    metadata = {
        "mode": "external_artifact",
        "path": str(prefix_source_path),
        "hpwl_threshold": None if hpwl_threshold is None else float(hpwl_threshold),
        "hpwl_threshold_raw": hpwl_threshold_raw,
        "loaded_rollout_count": int(len(payload.get("rollouts", []))),
        "eligible_rollout_count": int(len(rollouts)),
    }
    return rollouts, metadata


def run_ariane_reverse_curriculum_probe(
    *,
    output_path: Path,
    base_updates: int = 20,
    curriculum_updates: int = 20,
    candidate_rollouts: int = 20,
    top_k: int = 5,
    tail_steps: int = 10,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    prefix_source_path: Path | None = None,
    prefix_source_hpwl_threshold: float | None = None,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    _write_reverse_curriculum_status(
        output_path=output_path,
        status="running",
        stage="initializing",
        base_updates=int(base_updates),
        curriculum_updates=int(curriculum_updates),
        candidate_rollout_count=int(candidate_rollouts),
        tail_steps=int(tail_steps),
        grid_shape=list(resolved_grid_shape),
    )
    stage = "initializing"
    try:
        problem, policy, reward_model, adapter, metadata = _build_ariane_components(
            artifact_dir=artifact_dir,
            grid_shape=resolved_grid_shape,
            hidden_dim=hidden_dim,
        )
        checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
        load_policy_warm_start(policy, checkpoint)
        max_steps = len(adapter.ordered_macros())
        adapter_kwargs = {
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        }
        base_training = None
        candidate_rows: list[dict[str, object]] = []
        prefix_pool_states = []
        prefix_source_metadata: dict[str, object]

        if prefix_source_path is None:
            stage = "base_training"
            _write_reverse_curriculum_status(
                output_path=output_path,
                status="running",
                stage=stage,
                benchmark=metadata["benchmark"],
                warm_start_checkpoint=str(checkpoint),
            )
            base_reference_policy = _reference_policy_for_training(policy, training_config)
            base_training = train_ppo_policy(
                problem=problem,
                policy=policy,
                reference_policy=base_reference_policy,
                kl_anchor_beta=_training_kl_anchor_beta(training_config),
                grid_shape=resolved_grid_shape,
                reward_source="phase3_proxy_hpwl",
                max_steps=max_steps,
                num_updates=base_updates,
                learning_rate=float(training_config["learning_rate"]),
                seed=0,
                adapter_kwargs=adapter_kwargs,
                value_coef=float(training_config["value_coef"]),
                entropy_coef=float(training_config["entropy_coef"]),
                progress_every=max(1, base_updates // 5),
            )

            stage = "candidate_rollouts"
            _write_reverse_curriculum_status(
                output_path=output_path,
                status="running",
                stage=stage,
                base_final_hpwl=float(base_training["final_rollout"]["final_hpwl"]),
            )
            for rollout_seed in range(candidate_rollouts):
                rollout = collect_policy_rollout(
                    problem=problem,
                    policy=policy,
                    grid_shape=resolved_grid_shape,
                    reward_source="phase3_proxy_hpwl",
                    max_steps=max_steps,
                    seed=rollout_seed,
                    deterministic=False,
                    adapter_kwargs=adapter_kwargs,
                )
                row = {
                    "seed": rollout_seed,
                    "steps": int(rollout["steps"]),
                    "final_hpwl": float(rollout["final_hpwl"]),
                    "completed_episode": bool(rollout["completed_episode"]),
                    "termination_reason": str(rollout["termination_reason"]),
                }
                if not row["completed_episode"] and row["steps"] > tail_steps:
                    prefix_steps = int(row["steps"]) - tail_steps
                    prefix_state = _replay_prefix_state_from_rollout(
                        problem=problem,
                        reward_model=reward_model,
                        grid_shape=resolved_grid_shape,
                        rollout=rollout,
                        prefix_steps=prefix_steps,
                    )
                    row.update(
                        {
                            "prefix_steps": prefix_steps,
                            "remaining_steps": max_steps - prefix_steps,
                        }
                    )
                    prefix_pool_states.append((float(row["final_hpwl"]), prefix_state, row))
                candidate_rows.append(row)
            prefix_source_metadata = {
                "mode": "self_bootstrap",
                "loaded_rollout_count": int(candidate_rollouts),
                "eligible_rollout_count": int(len(prefix_pool_states)),
            }
        else:
            stage = "external_prefix_source"
            _write_reverse_curriculum_status(
                output_path=output_path,
                status="running",
                stage=stage,
                benchmark=metadata["benchmark"],
                warm_start_checkpoint=str(checkpoint),
                prefix_source_path=str(prefix_source_path),
            )
            loaded_rollouts, prefix_source_metadata = _load_external_prefix_candidates(
                prefix_source_path=prefix_source_path,
                hpwl_threshold=prefix_source_hpwl_threshold,
            )
            for rollout in loaded_rollouts:
                row = dict(rollout)
                row["steps"] = int(row["steps"])
                row["final_hpwl"] = float(row["final_hpwl"])
                row["completed_episode"] = bool(row["completed_episode"])
                row["termination_reason"] = str(row["termination_reason"])
                if row["completed_episode"] or row["steps"] <= tail_steps:
                    candidate_rows.append(row)
                    continue
                prefix_steps = int(row["steps"]) - tail_steps
                prefix_state = _replay_prefix_state_from_rollout(
                    problem=problem,
                    reward_model=reward_model,
                    grid_shape=resolved_grid_shape,
                    rollout=row,
                    prefix_steps=prefix_steps,
                )
                salvage = _evaluate_prefix_candidate_salvage(
                    problem=problem,
                    reward_model=reward_model,
                    grid_shape=resolved_grid_shape,
                    prefix_state=prefix_state,
                    reward_source="phase3_proxy_hpwl",
                )
                row.update(
                    {
                        "prefix_steps": prefix_steps,
                        "remaining_steps": max_steps - prefix_steps,
                        "salvage": salvage,
                    }
                )
                if bool(salvage["greedy"]["completed_episode"]):
                    prefix_pool_states.append((float(row["final_hpwl"]), prefix_state, row))
                candidate_rows.append(row)

        prefix_pool_states.sort(key=lambda item: item[0])
        selected_prefix_pool = prefix_pool_states[: max(1, top_k)]
        selected_prefix_rows = [dict(item[2]) for item in selected_prefix_pool]
        if not selected_prefix_pool:
            raise ValueError("Reverse curriculum probe found no salvageable prefix candidates")

        prefix_source_metadata["selected_prefix_pool_size"] = int(len(selected_prefix_rows))

        def curriculum_initial_state_factory(update_index: int):
            index = int(update_index) % len(selected_prefix_pool)
            return selected_prefix_pool[index][1]

        stage = "scratch_control"
        _write_reverse_curriculum_status(
            output_path=output_path,
            status="running",
            stage=stage,
            selected_prefix_pool_size=len(selected_prefix_rows),
            best_prefix_hpwl=float(selected_prefix_rows[0]["final_hpwl"]),
        )
        scratch_control_policy = _instantiate_ariane_warm_policy(
            problem=problem,
            grid_shape=resolved_grid_shape,
            hidden_dim=hidden_dim,
            checkpoint=checkpoint,
            fallback_policy=policy,
        )
        scratch_control_reference_policy = _reference_policy_for_training(scratch_control_policy, training_config)
        scratch_control = train_ppo_policy(
            problem=problem,
            policy=scratch_control_policy,
            reference_policy=scratch_control_reference_policy,
            kl_anchor_beta=_training_kl_anchor_beta(training_config),
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            num_updates=curriculum_updates,
            learning_rate=float(training_config["learning_rate"]),
            seed=1000,
            adapter_kwargs=adapter_kwargs,
            value_coef=float(training_config["value_coef"]),
            entropy_coef=float(training_config["entropy_coef"]),
            progress_every=max(1, curriculum_updates // 5),
            initial_state_factory=None,
        )

        stage = "reverse_curriculum"
        _write_reverse_curriculum_status(
            output_path=output_path,
            status="running",
            stage=stage,
            selected_prefix_pool_size=len(selected_prefix_rows),
            scratch_control_final_hpwl=float(scratch_control["final_rollout"]["final_hpwl"]),
        )
        curriculum_policy = _instantiate_ariane_warm_policy(
            problem=problem,
            grid_shape=resolved_grid_shape,
            hidden_dim=hidden_dim,
            checkpoint=checkpoint,
            fallback_policy=policy,
        )
        curriculum_reference_policy = _reference_policy_for_training(curriculum_policy, training_config)
        curriculum_training = train_ppo_policy(
            problem=problem,
            policy=curriculum_policy,
            reference_policy=curriculum_reference_policy,
            kl_anchor_beta=_training_kl_anchor_beta(training_config),
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            num_updates=curriculum_updates,
            learning_rate=float(training_config["learning_rate"]),
            seed=2000,
            adapter_kwargs=adapter_kwargs,
            value_coef=float(training_config["value_coef"]),
            entropy_coef=float(training_config["entropy_coef"]),
            progress_every=max(1, curriculum_updates // 5),
            initial_state_factory=curriculum_initial_state_factory,
        )

        artifact = {
            "benchmark": metadata["benchmark"],
            "metadata": {
                **metadata,
                "warm_start_checkpoint": str(checkpoint),
                "training_config": dict(training_config),
            },
            "base_updates": int(base_updates),
            "curriculum_updates": int(curriculum_updates),
            "candidate_rollout_count": int(candidate_rollouts),
            "tail_steps": int(tail_steps),
            "prefix_pool_size": len(selected_prefix_rows),
            "prefix_source": prefix_source_metadata,
            "prefix_candidates": candidate_rows,
            "selected_prefix_pool": selected_prefix_rows,
            "base_final_rollout": None if base_training is None else _rollout_json_summary(base_training["final_rollout"]),
            "scratch_control": {
                "history": scratch_control["history"],
                "final_rollout": _rollout_json_summary(scratch_control["final_rollout"]),
            },
            "reverse_curriculum": {
                "history": curriculum_training["history"],
                "final_rollout": _rollout_json_summary(curriculum_training["final_rollout"]),
            },
        }
        output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        _write_reverse_curriculum_status(
            output_path=output_path,
            status="completed",
            stage="completed",
            selected_prefix_pool_size=len(selected_prefix_rows),
            scratch_control_final_hpwl=float(scratch_control["final_rollout"]["final_hpwl"]),
            reverse_curriculum_final_hpwl=float(curriculum_training["final_rollout"]["final_hpwl"]),
        )
        return artifact
    except Exception as exc:
        _write_reverse_curriculum_status(
            output_path=output_path,
            status="failed",
            stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def export_ariane_prefix_pool(
    *,
    output_path: Path,
    num_updates: int = 100,
    candidate_rollouts: int = 20,
    policy_seeds: tuple[int, ...] = (0,),
    hpwl_threshold: float = 2.85,
    remaining_macro_range: tuple[int, int] = (8, 13),
    hpwl_threshold_sweep: tuple[float, ...] | None = None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None = None,
    geometry_packing_factor: float = 1.0,
    min_accepted_prefixes: int = 20,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    max_steps = len(adapter.ordered_macros())
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }
    base_training_runs: list[dict[str, object]] = []
    rollouts: list[dict[str, object]] = []
    for policy_seed in policy_seeds:
        policy = _instantiate_ariane_warm_policy(
            problem=problem,
            grid_shape=resolved_grid_shape,
            hidden_dim=hidden_dim,
            checkpoint=checkpoint,
            fallback_policy=policy,
        )
        reference_policy = _reference_policy_for_training(policy, training_config)
        training = train_ppo_policy(
            problem=problem,
            policy=policy,
            reference_policy=reference_policy,
            kl_anchor_beta=_training_kl_anchor_beta(training_config),
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            num_updates=num_updates,
            learning_rate=float(training_config["learning_rate"]),
            seed=int(policy_seed),
            adapter_kwargs=adapter_kwargs,
            value_coef=float(training_config["value_coef"]),
            entropy_coef=float(training_config["entropy_coef"]),
            progress_every=max(1, num_updates // 5),
        )
        base_training_runs.append(
            {
                "policy_seed": int(policy_seed),
                "final_rollout": _rollout_json_summary(training["final_rollout"]),
            }
        )
        for rollout_seed in range(candidate_rollouts):
            rollout = collect_policy_rollout(
                problem=problem,
                policy=policy,
                grid_shape=resolved_grid_shape,
                reward_source="phase3_proxy_hpwl",
                max_steps=max_steps,
                seed=rollout_seed,
                deterministic=False,
                adapter_kwargs=adapter_kwargs,
                policy_seed=int(policy_seed),
            )
            try:
                geometry_summary = _summarize_rollout_geometry_from_rollout(
                    adapter=adapter,
                    rollout=rollout,
                    grid_shape=resolved_grid_shape,
                )
            except Exception:
                geometry_summary = None
            acceptance = _evaluate_prefix_acceptance(
                rollout=rollout,
                geometry_summary=geometry_summary,
                hpwl_threshold=hpwl_threshold,
                remaining_macro_range=remaining_macro_range,
                geometry_packing_factor=geometry_packing_factor,
            )
            rollouts.append(
                {
                    "source_rollout_id": f"{int(policy_seed)}:{int(rollout_seed)}",
                    "policy_seed": int(policy_seed),
                    "seed": int(rollout_seed),
                    "steps": int(rollout["steps"]),
                    "final_hpwl": float(rollout["final_hpwl"]),
                    "completed_episode": bool(rollout["completed_episode"]),
                    "termination_reason": str(rollout["termination_reason"]),
                    "action_indices": _iter_rollout_action_indices(rollout),
                    "rollout_trace": _summarize_export_rollout_trace(
                        adapter=adapter,
                        rollout=rollout,
                        grid_shape=resolved_grid_shape,
                    ),
                    **acceptance,
                    "geometry_summary": geometry_summary,
                }
            )

    acceptance_summary = _summarize_prefix_acceptance(
        rollouts=rollouts,
        min_accepted_prefixes=min_accepted_prefixes,
        hpwl_threshold_sweep=hpwl_threshold_sweep,
        remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
        geometry_packing_factor=geometry_packing_factor,
    )

    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "warm_start_checkpoint": str(checkpoint),
            "training_config": dict(training_config),
        },
        "prefix_source": {
            "mode": "exported_policy_rollouts",
            "candidate_rollout_count": int(candidate_rollouts),
            "num_updates": int(num_updates),
            "policy_seed_count": int(len(policy_seeds)),
            "policy_seeds": [int(seed) for seed in policy_seeds],
        },
        "acceptance_criteria": {
            "hpwl_threshold": float(hpwl_threshold),
            "hpwl_threshold_raw": float(_normalize_hpwl_threshold(hpwl_threshold)),
            "remaining_macro_range": [int(remaining_macro_range[0]), int(remaining_macro_range[1])],
            "hpwl_threshold_sweep": [] if hpwl_threshold_sweep is None else [float(value) for value in hpwl_threshold_sweep],
            "hpwl_threshold_sweep_raw": [] if hpwl_threshold_sweep is None else [float(value) for value in _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep)],
            "remaining_macro_ranges_sweep": (
                []
                if remaining_macro_ranges_sweep is None
                else [[int(bounds[0]), int(bounds[1])] for bounds in remaining_macro_ranges_sweep]
            ),
            "geometry_packing_factor": float(geometry_packing_factor),
            "min_accepted_prefixes": int(min_accepted_prefixes),
        },
        "acceptance_summary": acceptance_summary,
        "base_final_rollout": base_training_runs[0]["final_rollout"] if base_training_runs else None,
        "base_training_runs": base_training_runs,
        "rollouts": rollouts,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def export_ariane_random_prefix_pool(
    *,
    output_path: Path,
    num_rollouts: int = 20,
    seeds: tuple[int, ...] | None = None,
    hpwl_threshold: float = 2.85,
    remaining_macro_range: tuple[int, int] = (8, 13),
    hpwl_threshold_sweep: tuple[float, ...] | None = None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None = None,
    geometry_packing_factor: float = 1.0,
    min_accepted_prefixes: int = 20,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    max_steps = len(adapter.ordered_macros())
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }
    rollout_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else tuple(range(num_rollouts))))
    rollouts: list[dict[str, object]] = []
    for rollout_seed in rollout_seeds:
        rollout = collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            seed=int(rollout_seed),
            deterministic=False,
            uniform_legal_policy=True,
            adapter_kwargs=adapter_kwargs,
        )
        try:
            geometry_summary = _summarize_rollout_geometry_from_rollout(
                adapter=adapter,
                rollout=rollout,
                grid_shape=resolved_grid_shape,
            )
        except Exception:
            geometry_summary = None
        acceptance = _evaluate_prefix_acceptance(
            rollout=rollout,
            geometry_summary=geometry_summary,
            hpwl_threshold=hpwl_threshold,
            remaining_macro_range=remaining_macro_range,
            geometry_packing_factor=geometry_packing_factor,
        )
        rollouts.append(
            {
                "source_rollout_id": f"random:{int(rollout_seed)}",
                "seed": int(rollout_seed),
                "steps": int(rollout["steps"]),
                "final_hpwl": float(rollout["final_hpwl"]),
                "completed_episode": bool(rollout["completed_episode"]),
                "termination_reason": str(rollout["termination_reason"]),
                "action_indices": _iter_rollout_action_indices(rollout),
                "rollout_trace": _summarize_export_rollout_trace(
                    adapter=adapter,
                    rollout=rollout,
                    grid_shape=resolved_grid_shape,
                ),
                **acceptance,
                "geometry_summary": geometry_summary,
            }
        )

    acceptance_summary = _summarize_prefix_acceptance(
        rollouts=rollouts,
        min_accepted_prefixes=min_accepted_prefixes,
        hpwl_threshold_sweep=hpwl_threshold_sweep,
        remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
        geometry_packing_factor=geometry_packing_factor,
    )
    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": metadata,
        "prefix_source": {
            "mode": "random_uniform_legal_rollouts",
            "num_rollouts": int(len(rollout_seeds)),
            "rollout_seeds": [int(seed) for seed in rollout_seeds],
        },
        "acceptance_criteria": {
            "hpwl_threshold": float(hpwl_threshold),
            "hpwl_threshold_raw": float(_normalize_hpwl_threshold(hpwl_threshold)),
            "remaining_macro_range": [int(remaining_macro_range[0]), int(remaining_macro_range[1])],
            "hpwl_threshold_sweep": [] if hpwl_threshold_sweep is None else [float(value) for value in hpwl_threshold_sweep],
            "hpwl_threshold_sweep_raw": [] if hpwl_threshold_sweep is None else [float(value) for value in _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep)],
            "remaining_macro_ranges_sweep": (
                []
                if remaining_macro_ranges_sweep is None
                else [[int(bounds[0]), int(bounds[1])] for bounds in remaining_macro_ranges_sweep]
            ),
            "geometry_packing_factor": float(geometry_packing_factor),
            "min_accepted_prefixes": int(min_accepted_prefixes),
        },
        "acceptance_summary": acceptance_summary,
        "distribution": {
            "final_hpwl": _summarize_scalar_distribution([float(row["final_hpwl"]) for row in rollouts]),
            "steps": _summarize_scalar_distribution([float(row["steps"]) for row in rollouts]),
            "threshold_hit_rates": _build_hpwl_threshold_hit_rates(
                rollouts=rollouts,
                hpwl_threshold_sweep=hpwl_threshold_sweep,
            ),
        },
        "rollouts": rollouts,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def export_ariane_warm_start_probe(
    *,
    output_path: Path,
    update0_rollouts: int = 25,
    update1_rollouts: int = 25,
    hpwl_threshold: float = 2.85,
    remaining_macro_range: tuple[int, int] = (8, 13),
    hpwl_threshold_sweep: tuple[float, ...] | None = None,
    remaining_macro_ranges_sweep: tuple[tuple[int, int], ...] | None = None,
    geometry_packing_factor: float = 1.0,
    min_accepted_prefixes: int = 20,
    tail_steps: int = 10,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    max_steps = len(adapter.ordered_macros())
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }

    update0_policy = _instantiate_ariane_warm_policy(
        problem=problem,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
        checkpoint=checkpoint,
        fallback_policy=policy,
    )
    update0_rows: list[dict[str, object]] = []
    for rollout_seed in range(update0_rollouts):
        rollout = collect_policy_rollout(
            problem=problem,
            policy=update0_policy,
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            seed=int(rollout_seed),
            deterministic=False,
            uniform_legal_policy=False,
            adapter_kwargs=adapter_kwargs,
        )
        update0_rows.append(
            _build_export_rollout_row(
                adapter=adapter,
                rollout=rollout,
                grid_shape=resolved_grid_shape,
                source_rollout_id=f"update0:{int(rollout_seed)}",
                hpwl_threshold=hpwl_threshold,
                remaining_macro_range=remaining_macro_range,
                geometry_packing_factor=geometry_packing_factor,
                extra_fields={"seed": int(rollout_seed), "phase": "update0"},
            )
        )

    update1_policy = initialize_policy_for_problem(problem=problem, grid_shape=resolved_grid_shape, hidden_dim=hidden_dim)
    load_policy_warm_start(update1_policy, checkpoint)
    update1_reference_policy = _reference_policy_for_training(update1_policy, training_config)
    update1_training = train_ppo_policy(
        problem=problem,
        policy=update1_policy,
        reference_policy=update1_reference_policy,
        kl_anchor_beta=_training_kl_anchor_beta(training_config),
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        num_updates=1,
        learning_rate=float(training_config["learning_rate"]),
        seed=int(training_config["init_seed"]),
        adapter_kwargs=adapter_kwargs,
        value_coef=float(training_config["value_coef"]),
        entropy_coef=float(training_config["entropy_coef"]),
        progress_every=1,
    )
    update1_rows: list[dict[str, object]] = []
    for rollout_seed in range(update1_rollouts):
        seed = 100 + int(rollout_seed)
        rollout = collect_policy_rollout(
            problem=problem,
            policy=update1_policy,
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=max_steps,
            seed=seed,
            deterministic=False,
            uniform_legal_policy=False,
            adapter_kwargs=adapter_kwargs,
        )
        update1_rows.append(
            _build_export_rollout_row(
                adapter=adapter,
                rollout=rollout,
                grid_shape=resolved_grid_shape,
                source_rollout_id=f"update1:{int(rollout_seed)}",
                hpwl_threshold=hpwl_threshold,
                remaining_macro_range=remaining_macro_range,
                geometry_packing_factor=geometry_packing_factor,
                extra_fields={"seed": seed, "phase": "update1"},
            )
        )

    all_rows = [*update0_rows, *update1_rows]
    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "warm_start_checkpoint": str(checkpoint),
            "training_config": dict(training_config),
        },
        "probe_config": {
            "update0_rollouts": int(update0_rollouts),
            "update1_rollouts": int(update1_rollouts),
            "tail_steps": int(tail_steps),
        },
        "acceptance_criteria": {
            "hpwl_threshold": float(hpwl_threshold),
            "hpwl_threshold_raw": float(_normalize_hpwl_threshold(hpwl_threshold)),
            "remaining_macro_range": [int(remaining_macro_range[0]), int(remaining_macro_range[1])],
            "hpwl_threshold_sweep": [] if hpwl_threshold_sweep is None else [float(value) for value in hpwl_threshold_sweep],
            "hpwl_threshold_sweep_raw": [] if hpwl_threshold_sweep is None else [float(value) for value in _normalize_hpwl_threshold_sweep(hpwl_threshold_sweep)],
            "remaining_macro_ranges_sweep": (
                []
                if remaining_macro_ranges_sweep is None
                else [[int(bounds[0]), int(bounds[1])] for bounds in remaining_macro_ranges_sweep]
            ),
            "geometry_packing_factor": float(geometry_packing_factor),
            "min_accepted_prefixes": int(min_accepted_prefixes),
        },
        "cohorts": {
            "update0": {
                **_summarize_rollout_probe_cohort(
                    rollouts=update0_rows,
                    hpwl_threshold_sweep=hpwl_threshold_sweep,
                    remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
                    geometry_packing_factor=geometry_packing_factor,
                    min_accepted_prefixes=min_accepted_prefixes,
                ),
                "rollouts": update0_rows,
            },
            "update1": {
                **_summarize_rollout_probe_cohort(
                    rollouts=update1_rows,
                    hpwl_threshold_sweep=hpwl_threshold_sweep,
                    remaining_macro_ranges_sweep=remaining_macro_ranges_sweep,
                    geometry_packing_factor=geometry_packing_factor,
                    min_accepted_prefixes=min_accepted_prefixes,
                ),
                "training_history": update1_training["history"],
                "final_rollout": _rollout_json_summary(update1_training["final_rollout"]),
                "rollouts": update1_rows,
            },
        },
        "salvage_calibration": _calibrate_rollout_rows_for_salvage(
            problem=problem,
            reward_model=reward_model,
            metadata=metadata,
            grid_shape=resolved_grid_shape,
            rollouts=all_rows,
            hpwl_threshold=hpwl_threshold,
            tail_steps=tail_steps,
        ),
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def calibrate_ariane_prefix_pool(
    *,
    output_path: Path,
    prefix_source_path: Path,
    tail_steps: int = 10,
    hpwl_threshold: float | None = None,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    problem, _, reward_model, _, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    loaded_rollouts, prefix_source_metadata = _load_external_prefix_candidates(
        prefix_source_path=prefix_source_path,
        hpwl_threshold=hpwl_threshold,
    )
    prefixes: list[dict[str, object]] = []
    greedy_completable = 0
    random_completable = 0
    for row in loaded_rollouts:
        if bool(row["completed_episode"]) or int(row["steps"]) <= int(tail_steps):
            continue
        prefix_steps = int(row["steps"]) - int(tail_steps)
        prefix_state = _replay_prefix_state_from_rollout(
            problem=problem,
            reward_model=reward_model,
            grid_shape=resolved_grid_shape,
            rollout=row,
            prefix_steps=prefix_steps,
        )
        salvage = _evaluate_prefix_candidate_salvage(
            problem=problem,
            reward_model=reward_model,
            grid_shape=resolved_grid_shape,
            prefix_state=prefix_state,
            reward_source="phase3_proxy_hpwl",
        )
        if bool(salvage["greedy"]["completed_episode"]):
            greedy_completable += 1
        if bool(salvage["random"]["completed_episode"]):
            random_completable += 1
        prefixes.append(
            {
                "name": row.get("name", f"prefix_{len(prefixes)}"),
                "steps": int(row["steps"]),
                "final_hpwl": float(row["final_hpwl"]),
                "prefix_steps": int(prefix_steps),
                "remaining_steps": int(metadata["ordered_macro_count"]) - int(prefix_steps),
                "salvage": salvage,
            }
        )

    recommended_mode = "greedy_or_random" if (greedy_completable > 0 or random_completable > 0) else "none"
    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": metadata,
        "prefix_source": prefix_source_metadata,
        "tail_steps": int(tail_steps),
        "prefixes": prefixes,
        "summary": {
            "total_candidates": int(len(prefixes)),
            "greedy_completable": int(greedy_completable),
            "random_completable": int(random_completable),
            "recommended_salvage_mode": recommended_mode,
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def _build_ariane_components(
    *,
    artifact_dir: Path,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    reward_weights: RewardWeights | None = None,
):
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    spec, problem, seed_placement, combined_lef_path = build_ariane_problem(artifact_dir=artifact_dir)
    reward_model, reward_metadata = build_ariane_reward_model(artifact_dir=artifact_dir, weights=reward_weights)
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=resolved_grid_shape,
    )
    policy = initialize_policy_for_problem(problem=problem, grid_shape=resolved_grid_shape, hidden_dim=hidden_dim)
    metadata = {
        "benchmark": spec.name,
        "grid_shape": list(resolved_grid_shape),
        "hidden_dim": hidden_dim,
        "combined_lef_path": str(combined_lef_path),
        "reference_def_path": str(spec.reference_def_path),
        "seed_macro_count": len(seed_placement),
        "ordered_macro_count": len(adapter.ordered_macros()),
        "reward_metadata": reward_metadata,
    }
    return problem, policy, reward_model, adapter, metadata


def _annotate_action_trace(
    *,
    adapter: ChipRLAdapter,
    action_trace: list[dict[str, object]],
    grid_shape: tuple[int, int],
) -> list[dict[str, object]]:
    state = adapter.initial_state()
    annotated: list[dict[str, object]] = []
    for row in action_trace:
        step_index = len(state.placed_macros)
        observation = adapter.observe(state, step_index=step_index)
        action = dict(row["action"]) if "action" in row else dict(row)
        legal_action_count = int((~observation["action_mask"]).sum().item())
        annotated.append(
            {
                "step_index": step_index,
                "macro": str(action["macro"]),
                "target": tuple(action["target"]),
                "legal_action_count": legal_action_count,
            }
        )
        state = adapter.apply_action(state, action)
    return annotated


def _trace_from_policy_rollout(
    *,
    adapter: ChipRLAdapter,
    rollout: dict[str, object],
    grid_shape: tuple[int, int],
) -> list[dict[str, object]]:
    trace: list[dict[str, object]] = []
    for transition in rollout["transitions"]:
        observation = transition.observation
        step_index = int(observation["step_index"])
        action_index = int(transition.action_index)
        row_index = action_index // grid_shape[1]
        col_index = action_index % grid_shape[1]
        trace.append(
            {
                "step_index": step_index,
                "macro": str(observation["current_macro"]),
                "target": adapter.grid_index_to_coordinate((row_index, col_index), grid_shape),
                "legal_action_count": int((~observation["action_mask"]).sum().item()),
            }
        )
    return trace


def _collect_greedy_episode_audit(
    *,
    problem,
    reward_model,
    grid_shape: tuple[int, int],
) -> dict[str, object]:
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=grid_shape,
    )
    greedy = run_greedy_episode(
        problem=problem,
        grid_shape=grid_shape,
        reward_source="phase3_proxy_hpwl",
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
    )
    trace = _annotate_action_trace(adapter=adapter, action_trace=greedy["trace"], grid_shape=grid_shape)
    return {
        **summarize_termination_audit(
            ordered_macros=adapter.ordered_macros(),
            trace=trace,
            termination_reason=str(greedy["termination_reason"]),
            final_hpwl=float(greedy["final_hpwl"]),
            episode_return=float(greedy["episode_return"]),
            is_legal=bool(greedy["is_legal"]),
        ),
    }


def _collect_policy_rollout_audit(
    *,
    problem,
    policy,
    reward_model,
    grid_shape: tuple[int, int],
    max_steps: int,
    seed: int,
) -> dict[str, object]:
    adapter = ChipRLAdapter(
        problem=problem,
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
        grid_shape=grid_shape,
    )
    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        seed=seed,
        deterministic=True,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
    )
    trace = _trace_from_policy_rollout(adapter=adapter, rollout=rollout, grid_shape=grid_shape)
    return {
        **summarize_termination_audit(
            ordered_macros=adapter.ordered_macros(),
            trace=trace,
            termination_reason=str(rollout["termination_reason"]),
            final_hpwl=float(rollout["final_hpwl"]),
            episode_return=float(rollout["episode_return"]),
            is_legal=bool(rollout["is_legal"]),
        ),
        "invalid_action_count": int(rollout["invalid_action_count"]),
    }


def zero_policy_parameters(policy: torch.nn.Module) -> None:
    for parameter in policy.parameters():
        parameter.data.zero_()


def run_uniform_policy_plumbing_check(
    *,
    output_path: Path,
    num_updates: int = 100,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    zero_policy_parameters(policy)
    training = train_ppo_policy(
        problem=problem,
        policy=policy,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        num_updates=num_updates,
        learning_rate=0.0,
        entropy_coef=0.0,
        seed=0,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
        uniform_legal_policy=True,
        progress_every=10,
    )
    history = training["history"]
    artifact = {
        "benchmark": metadata["benchmark"],
        "method": "uniform_policy_with_ppo_plumbing",
        "metadata": metadata,
        "num_updates": num_updates,
        "history": history,
        "final_rollout": {
            "steps": training["final_rollout"]["steps"],
            "episode_return": training["final_rollout"]["episode_return"],
            "final_hpwl": training["final_rollout"]["final_hpwl"],
            "invalid_action_count": training["final_rollout"]["invalid_action_count"],
            "termination_reason": training["final_rollout"]["termination_reason"],
            "completed_episode": training["final_rollout"]["completed_episode"],
        },
        "milestone_b1_gate": {
            "ran_requested_updates": len(history) == num_updates,
            "all_policy_losses_finite": all(torch.isfinite(torch.tensor(item["policy_loss"])) for item in history),
            "all_value_losses_finite": all(torch.isfinite(torch.tensor(item["value_loss"])) for item in history),
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_single_episode_overfit(
    *,
    output_path: Path,
    num_updates: int = 100,
    eval_every: int = 10,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    greedy_episode = run_greedy_episode(
        problem=problem,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        reward_model=reward_model,
        hpwl_evaluator=reward_model.hpwl_evaluator,
    )
    fixed_rollout = collect_rollout_for_action_trace(
        problem=problem,
        policy=policy,
        action_trace=greedy_episode["trace"],
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        seed=0,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
    )
    result = overfit_fixed_rollout(
        problem=problem,
        policy=policy,
        fixed_rollout=fixed_rollout,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        num_updates=num_updates,
        eval_every=eval_every,
        learning_rate=1e-3,
        seed=0,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
    )
    artifact = {
        "benchmark": metadata["benchmark"],
        "method": "fixed_episode_overfit",
        "metadata": metadata,
        "teacher_trace_length": len(greedy_episode["trace"]),
        "fixed_rollout": {
            "steps": len(fixed_rollout.transitions),
            "episode_return": fixed_rollout.episode_return,
            "final_hpwl": fixed_rollout.final_hpwl,
            "termination_reason": fixed_rollout.termination_reason,
            "completed_episode": fixed_rollout.completed_episode,
        },
        "initial_eval": {
            "episode_return": result["initial_eval"]["episode_return"],
            "final_hpwl": result["initial_eval"]["final_hpwl"],
            "termination_reason": result["initial_eval"]["termination_reason"],
            "completed_episode": result["initial_eval"]["completed_episode"],
        },
        "final_eval": {
            "episode_return": result["final_eval"]["episode_return"],
            "final_hpwl": result["final_eval"]["final_hpwl"],
            "termination_reason": result["final_eval"]["termination_reason"],
            "completed_episode": result["final_eval"]["completed_episode"],
        },
        "history": result["history"],
        "milestone_b1_gate": {
            "final_return_improved": result["final_eval"]["episode_return"] > result["initial_eval"]["episode_return"],
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_ppo_checkpoint(
    *,
    output_path: Path,
    num_updates: int = 2000,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    reward_weights: RewardWeights | None = None,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
        reward_weights=reward_weights,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    load_policy_warm_start(policy, checkpoint)
    metadata = {
        **metadata,
        "training_config": dict(training_config),
        "warm_start_checkpoint": str(checkpoint),
    }
    reference_policy = _reference_policy_for_training(policy, training_config)
    training = train_ppo_policy(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=_training_kl_anchor_beta(training_config),
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        num_updates=num_updates,
        learning_rate=float(training_config["learning_rate"]),
        seed=0,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
        value_coef=float(training_config["value_coef"]),
        entropy_coef=float(training_config["entropy_coef"]),
        progress_every=100,
    )
    history = training["history"]
    final_rollout = training["final_rollout"]
    random_mean_hpwl, greedy_final_hpwl, reference_warnings = _load_ariane_reference_levels(artifact_dir)
    final_hpwl = float(final_rollout["final_hpwl"])
    ppo_improvement_fraction = None
    greedy_progress_fraction = None
    first_update_reaching_beam_level = None
    first_update_reaching_80pct_of_greedy_gap = None
    best_gap_to_greedy = None
    best_gap_to_random = None
    if random_mean_hpwl is not None and random_mean_hpwl > 0:
        ppo_improvement_fraction = (random_mean_hpwl - final_hpwl) / random_mean_hpwl
        if greedy_final_hpwl is not None and random_mean_hpwl > greedy_final_hpwl:
            greedy_progress_fraction = (random_mean_hpwl - final_hpwl) / (random_mean_hpwl - greedy_final_hpwl)
    enriched_history = []
    for update_index, item in enumerate(history, start=1):
        current = dict(item)
        gap_to_random = None
        gap_to_greedy = None
        reaches_beam_level = None
        reaches_80pct_greedy_gap = None
        if random_mean_hpwl is not None:
            gap_to_random = float(current["final_hpwl"]) - random_mean_hpwl
        if greedy_final_hpwl is not None:
            gap_to_greedy = float(current["final_hpwl"]) - greedy_final_hpwl
            reaches_beam_level = float(current["final_hpwl"]) <= greedy_final_hpwl
            if reaches_beam_level and first_update_reaching_beam_level is None:
                first_update_reaching_beam_level = update_index
            if random_mean_hpwl is not None and random_mean_hpwl > greedy_final_hpwl:
                progress_fraction = (random_mean_hpwl - float(current["final_hpwl"])) / (random_mean_hpwl - greedy_final_hpwl)
                reaches_80pct_greedy_gap = progress_fraction >= 0.80
                if reaches_80pct_greedy_gap and first_update_reaching_80pct_of_greedy_gap is None:
                    first_update_reaching_80pct_of_greedy_gap = update_index
        if gap_to_greedy is not None:
            best_gap_to_greedy = gap_to_greedy if best_gap_to_greedy is None else min(best_gap_to_greedy, gap_to_greedy)
        if gap_to_random is not None:
            best_gap_to_random = gap_to_random if best_gap_to_random is None else min(best_gap_to_random, gap_to_random)
        current.update(
            {
                "gap_to_random": gap_to_random,
                "gap_to_greedy": gap_to_greedy,
                "reaches_beam_level": reaches_beam_level,
                "reaches_80pct_of_greedy_gap": reaches_80pct_greedy_gap,
            }
        )
        enriched_history.append(current)
    history = enriched_history
    legality_rate = float(mean(item["is_legal"] for item in history)) if history else 0.0
    completed_episode_rate = float(mean(item["completed_episode"] for item in history)) if history else 0.0
    entropy_final = float(history[-1]["entropy"]) if history else 0.0
    entropy_min = float(min(item["entropy"] for item in history)) if history else 0.0
    approx_kl_mean = float(mean(item["approx_kl"] for item in history)) if history else 0.0
    rollout_geometry = _summarize_rollout_geometry_from_rollout(
        adapter=adapter,
        rollout=final_rollout,
        grid_shape=resolved_grid_shape,
    )
    artifact = {
        "benchmark": metadata["benchmark"],
        "method": "ppo_scratch",
        "metadata": metadata,
        "num_updates": num_updates,
        "history": history,
        "history_step_histogram": _build_step_histogram(history),
        "reference_levels": {
            "masked_random_mean_hpwl": random_mean_hpwl,
            "greedy_final_hpwl": greedy_final_hpwl,
        },
        "reference_levels_status": {
            "available": random_mean_hpwl is not None and greedy_final_hpwl is not None,
            "warnings": reference_warnings,
        },
        "checkpoint_summary": {
            "final_hpwl": final_hpwl,
            "ppo_improvement_fraction": ppo_improvement_fraction,
            "greedy_progress_fraction": greedy_progress_fraction,
            "best_gap_to_random": best_gap_to_random,
            "best_gap_to_greedy": best_gap_to_greedy,
            "first_update_reaching_beam_level": first_update_reaching_beam_level,
            "first_update_reaching_80pct_of_greedy_gap": first_update_reaching_80pct_of_greedy_gap,
            "legality_rate": legality_rate,
            "completed_episode_rate": completed_episode_rate,
            "entropy_final": entropy_final,
            "entropy_min": entropy_min,
            "approx_kl_mean": approx_kl_mean,
            "remaining_macro_count": rollout_geometry["remaining_macro_count"],
        },
        "final_rollout": {
            "steps": final_rollout["steps"],
            "episode_return": final_rollout["episode_return"],
            "final_hpwl": final_rollout["final_hpwl"],
            "invalid_action_count": final_rollout["invalid_action_count"],
            "termination_reason": final_rollout["termination_reason"],
            "completed_episode": final_rollout["completed_episode"],
            "is_legal": final_rollout["is_legal"],
        },
        "final_rollout_geometry": rollout_geometry,
        "checkpoint_gate": {
            "beats_masked_random_30pct": ppo_improvement_fraction is not None and ppo_improvement_fraction >= 0.30,
            "reaches_80pct_of_greedy_gap": greedy_progress_fraction is not None and greedy_progress_fraction >= 0.80,
            "legality_rate_ge_95pct": legality_rate >= 0.95,
            "entropy_not_collapsed": entropy_min > 0.1,
            "reference_levels_available": random_mean_hpwl is not None and greedy_final_hpwl is not None,
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_warm_start_substep_diagnostic(
    *,
    output_path: Path,
    policy_epochs: int = 5,
    diagnostic_eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    reward_weights: RewardWeights | None = None,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
        reward_weights=reward_weights,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    load_policy_warm_start(policy, checkpoint)
    reference_policy = _reference_policy_for_training(policy, training_config)
    update_metrics = run_single_ppo_update(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=_training_kl_anchor_beta(training_config),
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        learning_rate=float(training_config["learning_rate"]),
        seed=int(training_config["init_seed"]),
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
        value_coef=float(training_config["value_coef"]),
        entropy_coef=float(training_config["entropy_coef"]),
        policy_epochs=int(policy_epochs),
        diagnostic_eval_seeds=tuple(int(seed) for seed in diagnostic_eval_seeds),
    )
    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "training_config": dict(training_config),
            "warm_start_checkpoint": str(checkpoint),
        },
        "diagnostic_config": {
            "policy_epochs": int(policy_epochs),
            "diagnostic_eval_seeds": [int(seed) for seed in diagnostic_eval_seeds],
        },
        "update_metrics": update_metrics,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def _rollout_to_fixed_rollout_batch(rollout: dict[str, object]) -> FixedRolloutBatch:
    return FixedRolloutBatch(
        transitions=rollout["transitions"],
        schema=rollout["schema"],
        final_hpwl=float(rollout["final_hpwl"]),
        episode_return=float(rollout["episode_return"]),
        termination_reason=str(rollout["termination_reason"]),
        completed_episode=bool(rollout["completed_episode"]),
    )


def _build_ariane_critic_prefit_comparison_conditions() -> dict[str, dict[str, float | str]]:
    return {
        "raw_baseline": {
            "prefit_learning_rate": 1e-3,
            "prefit_epochs": 20,
            "prefit_lr_warmup_epochs": 0,
            "target_transform": "raw",
        },
        "raw_fair_shot": {
            "prefit_learning_rate": 1e-3,
            "prefit_epochs": 200,
            "prefit_lr_warmup_epochs": 20,
            "target_transform": "raw",
        },
        "zscore_fair_shot": {
            "prefit_learning_rate": 1e-3,
            "prefit_epochs": 200,
            "prefit_lr_warmup_epochs": 20,
            "target_transform": "zscore",
        },
    }


def run_ariane_critic_prefit_comparison(
    *,
    output_path: Path,
    prefit_rollout_seeds: tuple[int, ...] = (0, 1, 2, 3),
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    reward_weights: RewardWeights | None = None,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    torch.manual_seed(0)
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
        reward_weights=reward_weights,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    load_policy_warm_start(policy, checkpoint)
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }

    prefit_rollouts = [
        collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=len(adapter.ordered_macros()),
            seed=int(seed),
            deterministic=False,
            adapter_kwargs=adapter_kwargs,
        )
        for seed in prefit_rollout_seeds
    ]
    fixed_rollouts = [_rollout_to_fixed_rollout_batch(rollout) for rollout in prefit_rollouts]
    simple_signal_diagnostic = summarize_rollout_return_signal(
        fixed_rollouts=fixed_rollouts,
        total_macro_count=len(adapter.ordered_macros()),
    )
    warm_start_rollout_summary = {
        "rollout_count": int(len(prefit_rollouts)),
        "completion_rate": float(
            sum(1 for rollout in prefit_rollouts if bool(rollout.get("completed_episode", False)))
            / max(1, len(prefit_rollouts))
        ),
        "final_hpwl_distribution": _summarize_scalar_distribution(
            [float(rollout["final_hpwl"]) for rollout in prefit_rollouts]
        ),
        "episode_return_distribution": _summarize_scalar_distribution(
            [float(rollout["episode_return"]) for rollout in prefit_rollouts]
        ),
    }

    conditions: dict[str, dict[str, object]] = {}
    prefit_gate: dict[str, dict[str, object]] = {}
    for condition_name, condition_config in _build_ariane_critic_prefit_comparison_conditions().items():
        condition_policy = copy.deepcopy(policy)
        metrics = prefit_value_head_from_rollouts(
            policy=condition_policy,
            fixed_rollouts=fixed_rollouts,
            learning_rate=float(condition_config["prefit_learning_rate"]),
            epochs=int(condition_config["prefit_epochs"]),
            lr_warmup_epochs=int(condition_config["prefit_lr_warmup_epochs"]),
            target_transform=str(condition_config["target_transform"]),
        )
        conditions[condition_name] = {
            **{key: value for key, value in condition_config.items()},
            **metrics,
        }
        correlation_gate = float(metrics.get("final_value_return_correlation", 0.0)) > 0.5
        prefit_gate[condition_name] = {
            "value_return_correlation_gt_0_5": bool(correlation_gate),
            "pass": bool(correlation_gate),
        }

    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "warm_start_checkpoint": str(checkpoint),
        },
        "diagnostic_config": {
            "prefit_rollout_seeds": [int(seed) for seed in prefit_rollout_seeds],
            "conditions": _build_ariane_critic_prefit_comparison_conditions(),
        },
        "warm_start_rollout_summary": warm_start_rollout_summary,
        "simple_signal_diagnostic": simple_signal_diagnostic,
        "conditions": conditions,
        "prefit_gate": prefit_gate,
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_critic_prefit_diagnostic(
    *,
    output_path: Path,
    prefit_rollout_seeds: tuple[int, ...] = (0, 1, 2, 3),
    diagnostic_eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    prefit_epochs: int = 20,
    policy_epochs: int = 1,
    followup_updates: int = 10,
    warm_start_baseline_hpwl: float = 2_881_800.0,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
    reward_weights: RewardWeights | None = None,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    warm_value_prefit_config = build_ariane_warm_value_prefit_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
        reward_weights=reward_weights,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    load_policy_warm_start(policy, checkpoint)
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }

    prefit_rollouts = [
        collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=len(adapter.ordered_macros()),
            seed=int(seed),
            deterministic=False,
            adapter_kwargs=adapter_kwargs,
        )
        for seed in prefit_rollout_seeds
    ]
    prefit_metrics = prefit_value_head_from_rollouts(
        policy=policy,
        fixed_rollouts=[_rollout_to_fixed_rollout_batch(rollout) for rollout in prefit_rollouts],
        learning_rate=float(warm_value_prefit_config["prefit_learning_rate"]),
        epochs=int(prefit_epochs),
    )

    reference_policy = _reference_policy_for_training(policy, training_config)
    update_metrics = run_single_ppo_update(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=_training_kl_anchor_beta(training_config),
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=len(adapter.ordered_macros()),
        learning_rate=float(training_config["learning_rate"]),
        seed=int(training_config["init_seed"]),
        adapter_kwargs=adapter_kwargs,
        value_coef=float(training_config["value_coef"]),
        entropy_coef=float(training_config["entropy_coef"]),
        policy_epochs=int(policy_epochs),
        diagnostic_eval_seeds=tuple(int(seed) for seed in diagnostic_eval_seeds),
    )

    substep_zero = update_metrics["substep_history"][0] if update_metrics.get("substep_history") else {}
    finite_kl = bool(
        "update_batch_kl_to_reference" in substep_zero
        and torch.isfinite(torch.tensor(float(substep_zero["update_batch_kl_to_reference"]))).item()
        and float(substep_zero["update_batch_kl_to_reference"]) < 1.0
    )
    heldout_log_prob_delta_within_band = bool(
        "heldout_log_prob_delta_mean" in substep_zero
        and abs(float(substep_zero["heldout_log_prob_delta_mean"])) < 0.05
    )
    necessary = {
        "finite_kl": finite_kl,
        "heldout_log_prob_delta_within_band": heldout_log_prob_delta_within_band,
        "pass": bool(finite_kl and heldout_log_prob_delta_within_band),
    }

    followup_training = None
    followup_training_artifact = None
    followup_eval = None
    sufficient = {
        "followup_hpwl_within_3pct_of_baseline": False,
        "pass": False,
    }
    if necessary["pass"] and int(followup_updates) > 0:
        followup_training = train_ppo_policy(
            problem=problem,
            policy=policy,
            reference_policy=reference_policy,
            kl_anchor_beta=_training_kl_anchor_beta(training_config),
            grid_shape=resolved_grid_shape,
            reward_source="phase3_proxy_hpwl",
            max_steps=len(adapter.ordered_macros()),
            num_updates=int(followup_updates),
            learning_rate=float(training_config["learning_rate"]),
            seed=int(training_config["init_seed"]),
            adapter_kwargs=adapter_kwargs,
            value_coef=float(training_config["value_coef"]),
            entropy_coef=float(training_config["entropy_coef"]),
        )
        followup_training_artifact = {
            "history": followup_training["history"],
            "final_rollout": _rollout_json_summary(followup_training["final_rollout"]),
        }
        followup_rollouts = [
            collect_policy_rollout(
                problem=problem,
                policy=policy,
                grid_shape=resolved_grid_shape,
                reward_source="phase3_proxy_hpwl",
                max_steps=len(adapter.ordered_macros()),
                seed=int(seed),
                deterministic=False,
                adapter_kwargs=adapter_kwargs,
            )
            for seed in diagnostic_eval_seeds
        ]
        hpwls = [float(rollout["final_hpwl"]) for rollout in followup_rollouts]
        followup_eval = {
            "rollout_count": int(len(followup_rollouts)),
            "final_hpwl_distribution": _summarize_scalar_distribution(hpwls),
        }
        median_hpwl = float(followup_eval["final_hpwl_distribution"]["median"])
        threshold = float(warm_start_baseline_hpwl) * 1.03
        sufficient["followup_hpwl_within_3pct_of_baseline"] = bool(median_hpwl <= threshold)
        sufficient["pass"] = bool(sufficient["followup_hpwl_within_3pct_of_baseline"])

    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "training_config": dict(training_config),
            "warm_value_prefit_config": dict(warm_value_prefit_config),
            "warm_start_checkpoint": str(checkpoint),
        },
        "diagnostic_config": {
            "prefit_rollout_seeds": [int(seed) for seed in prefit_rollout_seeds],
            "diagnostic_eval_seeds": [int(seed) for seed in diagnostic_eval_seeds],
            "prefit_epochs": int(prefit_epochs),
            "policy_epochs": int(policy_epochs),
            "followup_updates": int(followup_updates),
            "warm_start_baseline_hpwl": float(warm_start_baseline_hpwl),
        },
        "prefit_metrics": prefit_metrics,
        "update_metrics": update_metrics,
        "followup_training": followup_training_artifact,
        "followup_eval": followup_eval,
        "gate_status": {
            "necessary": necessary,
            "sufficient": sufficient,
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_b2_separated_compare(
    *,
    output_path: Path,
    num_updates: int = 200,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    problem, _, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    max_steps = len(adapter.ordered_macros())
    adapter_kwargs = {
        "reward_model": reward_model,
        "hpwl_evaluator": reward_model.hpwl_evaluator,
    }
    random_mean_hpwl, greedy_final_hpwl, reference_warnings = _load_ariane_reference_levels(artifact_dir)

    def init_policy(seed: int):
        torch.manual_seed(seed)
        return initialize_policy_for_problem(problem=problem, grid_shape=resolved_grid_shape, hidden_dim=hidden_dim)

    def summarize_run(name: str, training_config: dict[str, float], training: dict[str, object], prefit: dict[str, float] | None = None):
        final_rollout = training["final_rollout"]
        final_hpwl = float(final_rollout["final_hpwl"])
        gap_to_random = None if random_mean_hpwl is None else final_hpwl - random_mean_hpwl
        gap_to_greedy = None if greedy_final_hpwl is None else final_hpwl - greedy_final_hpwl
        return {
            "training_config": dict(training_config),
            "prefit": prefit,
            "final_rollout": {
                "steps": final_rollout["steps"],
                "episode_return": final_rollout["episode_return"],
                "final_hpwl": final_hpwl,
                "invalid_action_count": final_rollout["invalid_action_count"],
                "termination_reason": final_rollout["termination_reason"],
                "completed_episode": final_rollout["completed_episode"],
                "is_legal": final_rollout["is_legal"],
            },
            "history": training["history"],
            "summary": {
                "gap_to_random": gap_to_random,
                "gap_to_greedy": gap_to_greedy,
                "improvement_vs_random_fraction": None if random_mean_hpwl is None else (random_mean_hpwl - final_hpwl) / random_mean_hpwl,
                "progress_vs_greedy_fraction": None if (random_mean_hpwl is None or greedy_final_hpwl is None or random_mean_hpwl <= greedy_final_hpwl) else (random_mean_hpwl - final_hpwl) / (random_mean_hpwl - greedy_final_hpwl),
            },
        }

    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)

    scratch_config = build_ariane_scratch_training_config()
    scratch_policy = init_policy(int(scratch_config["init_seed"]))
    scratch_training = train_ppo_policy(
        problem=problem,
        policy=scratch_policy,
        grid_shape=grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        num_updates=num_updates,
        learning_rate=float(scratch_config["learning_rate"]),
        seed=0,
        adapter_kwargs=adapter_kwargs,
        value_coef=float(scratch_config["value_coef"]),
        entropy_coef=float(scratch_config["entropy_coef"]),
        progress_every=max(1, num_updates // 10),
    )

    bc_policy = init_policy(0)
    load_policy_warm_start(bc_policy, checkpoint)
    bc_rollout = collect_policy_rollout(
        problem=problem,
        policy=bc_policy,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        seed=0,
        deterministic=True,
        adapter_kwargs=adapter_kwargs,
    )

    warm_entropy_config = build_ariane_warm_entropy_training_config()
    warm_entropy_policy = init_policy(int(warm_entropy_config["init_seed"]))
    load_policy_warm_start(warm_entropy_policy, checkpoint)
    warm_entropy_training = train_ppo_policy(
        problem=problem,
        policy=warm_entropy_policy,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        num_updates=num_updates,
        learning_rate=float(warm_entropy_config["learning_rate"]),
        seed=0,
        adapter_kwargs=adapter_kwargs,
        value_coef=float(warm_entropy_config["value_coef"]),
        entropy_coef=float(warm_entropy_config["entropy_coef"]),
        progress_every=max(1, num_updates // 10),
    )

    warm_value_prefit_config = build_ariane_warm_value_prefit_training_config()
    warm_value_prefit_policy = init_policy(int(warm_value_prefit_config["init_seed"]))
    load_policy_warm_start(warm_value_prefit_policy, checkpoint)
    prefit_rollout = collect_policy_rollout(
        problem=problem,
        policy=warm_value_prefit_policy,
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        seed=0,
        deterministic=True,
        adapter_kwargs=adapter_kwargs,
    )
    prefit_batch = FixedRolloutBatch(
        transitions=prefit_rollout["transitions"],
        schema=prefit_rollout["schema"],
        final_hpwl=float(prefit_rollout["final_hpwl"]),
        episode_return=float(prefit_rollout["episode_return"]),
        termination_reason=str(prefit_rollout["termination_reason"]),
        completed_episode=bool(prefit_rollout["completed_episode"]),
    )
    prefit_metrics = prefit_value_head_from_rollout(
        policy=warm_value_prefit_policy,
        fixed_rollout=prefit_batch,
        learning_rate=float(warm_value_prefit_config["prefit_learning_rate"]),
        epochs=int(warm_value_prefit_config["prefit_epochs"]),
    )
    warm_value_prefit_training = train_ppo_policy(
        problem=problem,
        policy=warm_value_prefit_policy,
        grid_shape=grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        num_updates=num_updates,
        learning_rate=float(warm_value_prefit_config["learning_rate"]),
        seed=0,
        adapter_kwargs=adapter_kwargs,
        value_coef=float(warm_value_prefit_config["value_coef"]),
        entropy_coef=float(warm_value_prefit_config["entropy_coef"]),
        progress_every=max(1, num_updates // 10),
    )

    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": metadata,
        "num_updates": num_updates,
        "reference_levels": {
            "masked_random_mean_hpwl": random_mean_hpwl,
            "greedy_final_hpwl": greedy_final_hpwl,
            "bc_rollout_final_hpwl": float(bc_rollout["final_hpwl"]),
        },
        "reference_levels_status": {
            "available": random_mean_hpwl is not None and greedy_final_hpwl is not None,
            "warnings": reference_warnings,
        },
        "bc_rollout": {
            "steps": bc_rollout["steps"],
            "episode_return": bc_rollout["episode_return"],
            "final_hpwl": bc_rollout["final_hpwl"],
            "termination_reason": bc_rollout["termination_reason"],
            "completed_episode": bc_rollout["completed_episode"],
            "is_legal": bc_rollout["is_legal"],
        },
        "runs": {
            "scratch": summarize_run("scratch", scratch_config, scratch_training),
            "warm_entropy": summarize_run("warm_entropy", warm_entropy_config, warm_entropy_training),
            "warm_value_prefit": summarize_run("warm_value_prefit", warm_value_prefit_config, warm_value_prefit_training, prefit=prefit_metrics),
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def run_ariane_termination_audit(
    *,
    output_path: Path,
    num_updates: int = 2000,
    grid_shape: tuple[int, int] | None = None,
    hidden_dim: int = 64,
) -> dict[str, object]:
    artifact_dir = output_path.parent
    resolved_grid_shape = _resolve_ariane_grid_shape(grid_shape)
    training_config = build_ariane_checkpoint_training_config()
    torch.manual_seed(int(training_config["init_seed"]))
    problem, policy, reward_model, adapter, metadata = _build_ariane_components(
        artifact_dir=artifact_dir,
        grid_shape=resolved_grid_shape,
        hidden_dim=hidden_dim,
    )
    checkpoint = resolve_ariane_warm_start_checkpoint(artifact_dir)
    load_policy_warm_start(policy, checkpoint)
    reference_policy = _reference_policy_for_training(policy, training_config)
    max_steps = len(adapter.ordered_macros())
    training = train_ppo_policy(
        problem=problem,
        policy=policy,
        reference_policy=reference_policy,
        kl_anchor_beta=_training_kl_anchor_beta(training_config),
        grid_shape=resolved_grid_shape,
        reward_source="phase3_proxy_hpwl",
        max_steps=max_steps,
        num_updates=num_updates,
        learning_rate=float(training_config["learning_rate"]),
        seed=0,
        adapter_kwargs={
            "reward_model": reward_model,
            "hpwl_evaluator": reward_model.hpwl_evaluator,
        },
        value_coef=float(training_config["value_coef"]),
        entropy_coef=float(training_config["entropy_coef"]),
        progress_every=max(1, num_updates // 20),
    )
    greedy_audit = _collect_greedy_episode_audit(
        problem=problem,
        reward_model=reward_model,
        grid_shape=resolved_grid_shape,
    )
    ppo_audit = _collect_policy_rollout_audit(
        problem=problem,
        policy=policy,
        reward_model=reward_model,
        grid_shape=resolved_grid_shape,
        max_steps=max_steps,
        seed=0,
    )
    divergence = find_first_trace_divergence(
        reference_trace=greedy_audit["trace"],
        candidate_trace=ppo_audit["trace"],
    )
    ppo_steps = int(ppo_audit.get("steps", ppo_audit.get("placed_macro_count", 0)))
    greedy_steps = int(greedy_audit.get("steps", greedy_audit.get("placed_macro_count", 0)))
    artifact = {
        "benchmark": metadata["benchmark"],
        "metadata": {
            **metadata,
            "training_config": dict(training_config),
            "warm_start_checkpoint": str(checkpoint),
        },
        "num_updates": num_updates,
        "training_history_tail": training["history"][-10:],
        "greedy": greedy_audit,
        "ppo": ppo_audit,
        "trace_divergence": divergence,
        "termination_comparison": {
            "ppo_minus_greedy_steps": ppo_steps - greedy_steps,
            "ppo_minus_greedy_hpwl": float(ppo_audit["final_hpwl"]) - float(greedy_audit["final_hpwl"]),
            "ppo_failed_early": bool(ppo_steps < greedy_steps),
        },
    }
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Ariane macroplacement PPO sanity checks.")
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "experiments" / "chip"),
    )
    parser.add_argument(
        "--run",
        choices=("plumbing", "overfit", "train", "audit", "all"),
        default="all",
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--grid-rows", type=int, default=32)
    parser.add_argument("--grid-cols", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_shape = (args.grid_rows, args.grid_cols)

    if args.run in {"plumbing", "all"}:
        run_uniform_policy_plumbing_check(
            output_path=output_dir / "phase3_uniform_policy_plumbing_ariane.json",
            num_updates=args.updates,
            grid_shape=grid_shape,
            hidden_dim=args.hidden_dim,
        )
    if args.run in {"overfit", "all"}:
        run_single_episode_overfit(
            output_path=output_dir / "phase3_overfit_single_episode_ariane.json",
            num_updates=args.updates,
            eval_every=args.eval_every,
            grid_shape=grid_shape,
            hidden_dim=args.hidden_dim,
        )
    if args.run in {"train", "all"}:
        run_ariane_ppo_checkpoint(
            output_path=output_dir / "phase3_ppo_ariane_checkpoint.json",
            num_updates=args.updates,
            grid_shape=grid_shape,
            hidden_dim=args.hidden_dim,
        )
    if args.run in {"audit", "all"}:
        run_ariane_termination_audit(
            output_path=output_dir / "phase3_ariane_termination_audit.json",
            num_updates=args.updates,
            grid_shape=grid_shape,
            hidden_dim=args.hidden_dim,
        )


if __name__ == "__main__":
    main()
