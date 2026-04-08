from __future__ import annotations

import json
from pathlib import Path

from chip.experiments.phase3_macroplacement_benchmarks import resolve_ariane_benchmark
from chip.experiments.phase3_macroplacement_orfs import (
    build_reference_orfs_completion_command,
    collect_reference_orfs_completion_metrics,
)


def macroplacement_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tmp" / "MacroPlacement"


def test_build_reference_orfs_completion_command_uses_floorplan_def_and_skip_cts_repair() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")

    command, paths = build_reference_orfs_completion_command(spec, workspace_root=workspace_root)

    joined = " ".join(command)
    assert command[:4] == ["docker", "run", "--rm", "-v"]
    assert "DESIGN_CONFIG=./designs/nangate45/ariane133/config.mk" in joined
    assert "FLOORPLAN_DEF=/workspace/tmp/MacroPlacement/Flows/NanGate45/ariane133/def/ariane133_fp_placed_macros.def" in joined
    assert "CORE_UTILIZATION=" in joined
    assert "SKIP_CTS_REPAIR_TIMING=1" in joined
    assert joined.endswith(" finish")
    assert paths.final_def_path.name == "6_final.def"


def test_collect_reference_orfs_completion_metrics_parses_orfs_outputs(tmp_path: Path) -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")
    flow_variant = "unit"
    result_dir = tmp_path / "results" / "nangate45" / "ariane133" / flow_variant
    log_dir = tmp_path / "logs" / "nangate45" / "ariane133" / flow_variant
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "clock_period.txt").write_text("3.0")
    final_def_path = result_dir / "6_final.def"
    final_def_path.write_text("VERSION 5.8 ;")
    (result_dir / "6_final.odb").write_text("odb")

    (log_dir / "6_report.json").write_text(
        json.dumps(
            {
                "finish__timing__setup__ws": -0.68,
                "finish__timing__setup__tns": -1473.26,
                "finish__timing__hold__ws": 0.02,
                "finish__timing__hold__tns": 0.0,
                "finish__design__instance__count__macros": 132,
                "finish__design__instance__count__stdcell": 187049,
                "finish__flow__warnings__count": 3,
                "finish__flow__errors__count": 0,
            }
        )
    )
    (log_dir / "5_2_route.json").write_text(
        json.dumps(
            {
                "detailedroute__route__wirelength": 7751289,
                "detailedroute__route__drc_errors": 0,
            }
        )
    )
    (log_dir / "4_1_cts.json").write_text(json.dumps({"cts__route__wirelength__estimated": 6853890}))
    (log_dir / "3_3_place_gp.json").write_text(json.dumps({"globalplace__route__wirelength__estimated": 6293205}))

    from chip.experiments import phase3_macroplacement_orfs as orfs_module

    original_materialize = orfs_module.materialize_combined_lef
    original_load = orfs_module.load_chip_problem
    original_hpwl = orfs_module.hpwl_cost
    orfs_module.materialize_combined_lef = lambda spec, output_dir: output_dir / "combined.lef"
    orfs_module.load_chip_problem = lambda lef, def_path: (None, object())
    orfs_module.hpwl_cost = lambda problem: 4884882.0
    try:
        metrics = collect_reference_orfs_completion_metrics(
            spec,
            work_home=tmp_path,
            flow_variant=flow_variant,
            runtime_seconds=123.4,
            skip_cts_repair_timing=True,
        )
    finally:
        orfs_module.materialize_combined_lef = original_materialize
        orfs_module.load_chip_problem = original_load
        orfs_module.hpwl_cost = original_hpwl

    assert metrics["placement_source"] == "ariane_reference_floorplan_orfs_completion"
    assert metrics["final_hpwl"] == 4884882.0
    assert metrics["total_routed_wirelength"] == 7751289
    assert metrics["drc_violations"] == 0
    assert metrics["wns"] == -0.68
    assert metrics["tns"] == -1473.26
    assert metrics["runtime_seconds"] == 123.4
    assert metrics["runtime_source"] == "wall_clock"
    assert metrics["skip_cts_repair_timing"] is True


def test_collect_reference_orfs_completion_metrics_allows_missing_global_place_wirelength(tmp_path: Path) -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")
    flow_variant = "unit_missing_gp"
    result_dir = tmp_path / "results" / "nangate45" / "ariane133" / flow_variant
    log_dir = tmp_path / "logs" / "nangate45" / "ariane133" / flow_variant
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    (result_dir / "clock_period.txt").write_text("3.0")
    (result_dir / "6_final.def").write_text("VERSION 5.8 ;")
    (result_dir / "6_final.odb").write_text("odb")
    (log_dir / "6_report.json").write_text(
        json.dumps(
            {
                "finish__timing__setup__ws": -0.68,
                "finish__timing__setup__tns": -1473.26,
                "finish__timing__hold__ws": 0.02,
                "finish__timing__hold__tns": 0.0,
                "finish__design__instance__count__macros": 132,
                "finish__design__instance__count__stdcell": 187049,
                "finish__flow__warnings__count": 3,
                "finish__flow__errors__count": 0,
            }
        )
    )
    (log_dir / "5_2_route.json").write_text(
        json.dumps(
            {
                "detailedroute__route__wirelength": 7751289,
                "detailedroute__route__drc_errors": 0,
            }
        )
    )
    (log_dir / "4_1_cts.json").write_text(json.dumps({"cts__route__wirelength__estimated": 6853890}))
    (log_dir / "3_3_place_gp.json").write_text(json.dumps({"globalplace__timing__setup__ws": -0.9}))

    from chip.experiments import phase3_macroplacement_orfs as orfs_module

    original_materialize = orfs_module.materialize_combined_lef
    original_load = orfs_module.load_chip_problem
    original_hpwl = orfs_module.hpwl_cost
    orfs_module.materialize_combined_lef = lambda spec, output_dir: output_dir / "combined.lef"
    orfs_module.load_chip_problem = lambda lef, def_path: (None, object())
    orfs_module.hpwl_cost = lambda problem: 4884882.0
    try:
        metrics = collect_reference_orfs_completion_metrics(
            spec,
            work_home=tmp_path,
            flow_variant=flow_variant,
            runtime_seconds=123.4,
            skip_cts_repair_timing=True,
        )
    finally:
        orfs_module.materialize_combined_lef = original_materialize
        orfs_module.load_chip_problem = original_load
        orfs_module.hpwl_cost = original_hpwl

    assert metrics["estimated_wirelength_after_global_place"] is None
