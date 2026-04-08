from __future__ import annotations

import os
import time
from pathlib import Path

from chip.experiments.phase3_macroplacement_benchmarks import (
    MacroPlacementBenchmarkSpec,
    materialize_combined_lef,
)
from chip.netlist_parser import load_chip_problem
from chip.openroad_eval import build_evaluation_request, evaluate_openroad_request
from chip.rl_adapter import ChipRLAdapter


def evaluate_reference_placement(
    spec: MacroPlacementBenchmarkSpec,
    *,
    artifact_dir: Path | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    resolved_artifact_dir = (artifact_dir or (spec.flow_dir / ".codex_probe")).resolve()
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    combined_lef_path = materialize_combined_lef(spec, output_dir=resolved_artifact_dir)
    _, problem = load_chip_problem(combined_lef_path, spec.reference_def_path)
    placement = {macro.name: (macro.x, macro.y) for macro in problem.macros.values() if macro.kind != "PIN"}
    orientations = {macro.name: macro.orientation for macro in problem.macros.values() if macro.kind != "PIN"}
    request = build_evaluation_request(
        benchmark_name=spec.name,
        lef_path=_relative_to_artifact_dir(spec.tech_lef_path, resolved_artifact_dir),
        lef_paths=tuple(_relative_to_artifact_dir(path, resolved_artifact_dir) for path in spec.lef_paths),
        def_path=_relative_to_artifact_dir(spec.reference_def_path, resolved_artifact_dir),
        placement_mode="sa_injected",
        artifact_dir=str(resolved_artifact_dir),
        liberty_path=_relative_to_artifact_dir(spec.standard_liberty_path, resolved_artifact_dir),
        sdc_path=_relative_to_artifact_dir(spec.sdc_path, resolved_artifact_dir),
        script_stem=f"{spec.name}_reference_eval",
        run_detailed_route=False,
    )
    metrics = evaluate_openroad_request(
        request,
        placement=placement,
        orientations=orientations,
        timeout_seconds=timeout_seconds,
    )
    metrics.update(
        {
            "benchmark": spec.name,
            "placement_mode": "reference_def",
            "reference_def_path": str(spec.reference_def_path),
            "combined_lef_path": str(combined_lef_path),
            "liberty_path": str(spec.standard_liberty_path),
        }
    )
    return metrics


def run_legality_probe(
    spec: MacroPlacementBenchmarkSpec,
    *,
    artifact_dir: Path | None = None,
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, object]:
    resolved_artifact_dir = (artifact_dir or (spec.flow_dir / ".codex_probe")).resolve()
    resolved_artifact_dir.mkdir(parents=True, exist_ok=True)
    combined_lef_path = materialize_combined_lef(spec, output_dir=resolved_artifact_dir)
    _, problem = load_chip_problem(combined_lef_path, spec.reference_def_path)
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    initial_state = adapter.initial_state()
    current_macro = adapter.current_macro_for_step(0)
    if current_macro is None:
        raise ValueError(f"No movable macros found for benchmark {spec.name}")

    target = adapter.grid_index_to_coordinate((grid_shape[0] // 2, grid_shape[1] // 2), grid_shape)
    legality_start = time.perf_counter()
    legality_state = adapter.apply_action(initial_state, {"macro": current_macro, "target": target})
    legality_report = adapter.constraint_report(legality_state)
    single_legality_check_seconds = time.perf_counter() - legality_start

    mask_start = time.perf_counter()
    action_mask = adapter.build_action_mask(initial_state, step_index=0, grid_shape=grid_shape)
    build_action_mask_seconds = time.perf_counter() - mask_start

    movable_macros = [macro for macro in problem.macros.values() if not macro.is_fixed and macro.kind != "PIN"]
    valid_action_fraction = float((~action_mask).sum().item()) / float(action_mask.numel())
    unique_macro_sizes = sorted({(macro.width, macro.height) for macro in movable_macros})

    return {
        "benchmark": spec.name,
        "variant": spec.variant,
        "platform": spec.platform,
        "reference_def_path": str(spec.reference_def_path),
        "combined_lef_path": str(combined_lef_path),
        "movable_macro_count": len(movable_macros),
        "fixed_object_count": sum(1 for macro in problem.macros.values() if macro.is_fixed),
        "net_count": len(problem.nets),
        "grid_shape": grid_shape,
        "single_legality_check_seconds": single_legality_check_seconds,
        "single_legality_check_valid": legality_report.is_valid,
        "build_action_mask_seconds": build_action_mask_seconds,
        "valid_action_fraction": valid_action_fraction,
        "unique_macro_size_count": len(unique_macro_sizes),
        "largest_macro_area": max((macro.width * macro.height for macro in movable_macros), default=0.0),
    }


def _relative_to_artifact_dir(path: Path, artifact_dir: Path) -> str:
    return os.path.relpath(path.resolve(), artifact_dir.resolve()).replace("\\", "/")
