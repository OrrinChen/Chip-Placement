from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from chip.cost import hpwl_cost
from chip.experiments.phase3_macroplacement_benchmarks import (
    MacroPlacementBenchmarkSpec,
    materialize_combined_lef,
)
from chip.netlist_parser import load_chip_problem


@dataclass(frozen=True)
class ORFSCompletionPaths:
    work_home: Path
    result_dir: Path
    log_dir: Path
    report_dir: Path
    final_def_path: Path
    final_odb_path: Path
    finish_json_path: Path
    route_json_path: Path
    cts_json_path: Path
    global_place_json_path: Path


def default_orfs_work_home(
    spec: MacroPlacementBenchmarkSpec,
    *,
    workspace_root: Path | None = None,
) -> Path:
    root = (workspace_root or _default_workspace_root()).resolve()
    return root / "tmp" / f"orfs_{spec.variant}_ref"


def build_reference_orfs_completion_command(
    spec: MacroPlacementBenchmarkSpec,
    *,
    workspace_root: Path | None = None,
    work_home: Path | None = None,
    flow_variant: str = "codex_ref",
    docker_image: str = "openroad/orfs",
    skip_cts_repair_timing: bool = True,
    target: str = "finish",
) -> tuple[list[str], ORFSCompletionPaths]:
    resolved_workspace_root = (workspace_root or _default_workspace_root()).resolve()
    resolved_work_home = (work_home or default_orfs_work_home(spec, workspace_root=resolved_workspace_root)).resolve()
    paths = _orfs_completion_paths(spec, work_home=resolved_work_home, flow_variant=flow_variant)

    workspace_mount = _windows_path_to_docker(resolved_workspace_root)
    work_home_in_container = _path_in_workspace(resolved_work_home, resolved_workspace_root)
    floorplan_def_in_container = _path_in_workspace(spec.reference_def_path.resolve(), resolved_workspace_root)
    platform_slug = spec.platform.lower()

    make_args = [
        f"DESIGN_CONFIG=./designs/{platform_slug}/{spec.variant}/config.mk",
        f"FLOW_VARIANT={flow_variant}",
        f"WORK_HOME={work_home_in_container}",
        f"FLOORPLAN_DEF={floorplan_def_in_container}",
        "CORE_UTILIZATION=",
    ]
    if skip_cts_repair_timing:
        make_args.append("SKIP_CTS_REPAIR_TIMING=1")
    make_args.append(target)
    make_invocation = " ".join(["make", *make_args])
    script = f"mkdir -p {work_home_in_container} && cd /OpenROAD-flow-scripts/flow && {make_invocation}"

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace_mount}:/workspace",
        docker_image,
        "bash",
        "-lc",
        script,
    ]
    return command, paths


def collect_reference_orfs_completion_metrics(
    spec: MacroPlacementBenchmarkSpec,
    *,
    work_home: Path | None = None,
    flow_variant: str = "codex_ref",
    runtime_seconds: float | None = None,
    skip_cts_repair_timing: bool = True,
) -> dict[str, object]:
    paths = _orfs_completion_paths(
        spec,
        work_home=(work_home or default_orfs_work_home(spec)).resolve(),
        flow_variant=flow_variant,
    )
    required = [
        paths.final_def_path,
        paths.finish_json_path,
        paths.route_json_path,
        paths.cts_json_path,
        paths.global_place_json_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing ORFS completion artifacts: {missing_paths}")

    finish_metrics = json.loads(paths.finish_json_path.read_text())
    route_metrics = json.loads(paths.route_json_path.read_text())
    cts_metrics = json.loads(paths.cts_json_path.read_text())
    global_place_metrics = json.loads(paths.global_place_json_path.read_text())
    estimated_wirelength_after_global_place = _first_present_metric(
        global_place_metrics,
        "globalplace__route__wirelength__estimated",
        "globalplace__route__wirelength",
    )

    combined_lef_path = materialize_combined_lef(spec, output_dir=paths.work_home / "artifacts")
    _, problem = load_chip_problem(combined_lef_path, paths.final_def_path)
    final_hpwl = hpwl_cost(problem)

    measured_runtime_seconds = runtime_seconds
    runtime_source = "wall_clock"
    if measured_runtime_seconds is None:
        measured_runtime_seconds = _timestamp_runtime_seconds(paths)
        runtime_source = "file_timestamps"

    return {
        "benchmark": spec.name,
        "variant": spec.variant,
        "platform": spec.platform,
        "placement_source": "ariane_reference_floorplan_orfs_completion",
        "reference_def_path": str(spec.reference_def_path),
        "final_def_path": str(paths.final_def_path),
        "final_odb_path": str(paths.final_odb_path),
        "orfs_work_home": str(paths.work_home),
        "orfs_result_dir": str(paths.result_dir),
        "orfs_log_dir": str(paths.log_dir),
        "flow_variant": flow_variant,
        "skip_cts_repair_timing": skip_cts_repair_timing,
        "orfs_completion_mode": "fast" if skip_cts_repair_timing else "full",
        "runtime_seconds": measured_runtime_seconds,
        "runtime_minutes": measured_runtime_seconds / 60.0,
        "runtime_source": runtime_source,
        "final_hpwl": final_hpwl,
        "estimated_wirelength_after_global_place": estimated_wirelength_after_global_place,
        "estimated_wirelength_after_cts": cts_metrics["cts__route__wirelength__estimated"],
        "total_routed_wirelength": route_metrics["detailedroute__route__wirelength"],
        "routing_overhead_ratio": route_metrics["detailedroute__route__wirelength"] / final_hpwl,
        "drc_violations": route_metrics["detailedroute__route__drc_errors"],
        "wns": finish_metrics["finish__timing__setup__ws"],
        "tns": finish_metrics["finish__timing__setup__tns"],
        "hold_wns": finish_metrics["finish__timing__hold__ws"],
        "hold_tns": finish_metrics["finish__timing__hold__tns"],
        "timing_gated": False,
        "timing_gate_note": (
            "WNS/TNS are reported but not gated; SKIP_CTS_REPAIR_TIMING=1 is enabled for fast, repeatable completion."
        ),
        "recommended_rl_reward_role": "offline_eval_only",
        "reward_role_note": (
            "ORFS completion wall time is too high for PPO inner-loop reward; use this path for offline final evaluation only."
        ),
        "macro_count": finish_metrics["finish__design__instance__count__macros"],
        "stdcell_count": finish_metrics["finish__design__instance__count__stdcell"],
        "finish_warning_count": finish_metrics["finish__flow__warnings__count"],
        "finish_error_count": finish_metrics["finish__flow__errors__count"],
    }


def run_reference_orfs_completion(
    spec: MacroPlacementBenchmarkSpec,
    *,
    workspace_root: Path | None = None,
    work_home: Path | None = None,
    flow_variant: str = "codex_ref",
    docker_image: str = "openroad/orfs",
    skip_cts_repair_timing: bool = True,
    target: str = "finish",
    force_rerun: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, object]:
    resolved_workspace_root = (workspace_root or _default_workspace_root()).resolve()
    command, paths = build_reference_orfs_completion_command(
        spec,
        workspace_root=resolved_workspace_root,
        work_home=work_home,
        flow_variant=flow_variant,
        docker_image=docker_image,
        skip_cts_repair_timing=skip_cts_repair_timing,
        target=target,
    )

    runtime_seconds: float | None = None
    if force_rerun or not paths.final_def_path.exists():
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=resolved_workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime_seconds = time.perf_counter() - started
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                returncode=completed.returncode,
                cmd=command,
                output=completed.stdout,
                stderr=completed.stderr,
            )

    metrics = collect_reference_orfs_completion_metrics(
        spec,
        work_home=paths.work_home,
        flow_variant=flow_variant,
        runtime_seconds=runtime_seconds,
        skip_cts_repair_timing=skip_cts_repair_timing,
    )
    metrics["fresh_run"] = runtime_seconds is not None
    return metrics


def _orfs_completion_paths(
    spec: MacroPlacementBenchmarkSpec,
    *,
    work_home: Path,
    flow_variant: str,
) -> ORFSCompletionPaths:
    platform_slug = spec.platform.lower()
    result_dir = work_home / "results" / platform_slug / spec.variant / flow_variant
    log_dir = work_home / "logs" / platform_slug / spec.variant / flow_variant
    report_dir = work_home / "reports" / platform_slug / spec.variant / flow_variant
    return ORFSCompletionPaths(
        work_home=work_home,
        result_dir=result_dir,
        log_dir=log_dir,
        report_dir=report_dir,
        final_def_path=result_dir / "6_final.def",
        final_odb_path=result_dir / "6_final.odb",
        finish_json_path=log_dir / "6_report.json",
        route_json_path=log_dir / "5_2_route.json",
        cts_json_path=log_dir / "4_1_cts.json",
        global_place_json_path=log_dir / "3_3_place_gp.json",
    )


def _timestamp_runtime_seconds(paths: ORFSCompletionPaths) -> float:
    start_candidates = [
        paths.result_dir / "clock_period.txt",
        paths.result_dir / "1_1_yosys_canonicalize.rtlil",
        paths.result_dir / "1_synth.odb",
    ]
    end_candidates = [
        paths.final_def_path,
        paths.final_odb_path,
        paths.finish_json_path,
    ]
    existing_starts = [path.stat().st_mtime for path in start_candidates if path.exists()]
    existing_ends = [path.stat().st_mtime for path in end_candidates if path.exists()]
    if not existing_starts or not existing_ends:
        raise FileNotFoundError("Unable to infer ORFS runtime from timestamps")
    return max(existing_ends) - min(existing_starts)


def _first_present_metric(metrics: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _path_in_workspace(path: Path, workspace_root: Path) -> str:
    relative = path.resolve().relative_to(workspace_root.resolve())
    return f"/workspace/{relative.as_posix()}"


def _windows_path_to_docker(path: Path) -> str:
    return path.resolve().as_posix()


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]
