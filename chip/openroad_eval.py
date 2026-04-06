from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from chip.openroad_bridge import OpenROADBridge


@dataclass(frozen=True)
class EvaluationRequest:
    benchmark_name: str
    lef_path: str
    def_path: str
    placement_mode: str
    artifact_dir: str = "."
    lef_paths: tuple[str, ...] = ()
    liberty_path: str | None = None
    sdc_path: str | None = None
    detailed_route_end_iter: int = 0
    placement_density: float | None = None
    write_def_path: str | None = None


def build_evaluation_request(
    benchmark_name: str,
    lef_path: str,
    def_path: str,
    placement_mode: str,
    artifact_dir: str = ".",
    lef_paths: tuple[str, ...] | None = None,
    liberty_path: str | None = None,
    sdc_path: str | None = None,
    detailed_route_end_iter: int = 0,
    placement_density: float | None = None,
    write_def_path: str | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        benchmark_name=benchmark_name,
        lef_path=lef_path,
        def_path=def_path,
        placement_mode=placement_mode,
        artifact_dir=artifact_dir,
        lef_paths=lef_paths or (),
        liberty_path=liberty_path,
        sdc_path=sdc_path,
        detailed_route_end_iter=detailed_route_end_iter,
        placement_density=placement_density,
        write_def_path=write_def_path,
    )


def parse_openroad_metrics(stdout: str) -> dict[str, float | int]:
    def extract_float(pattern: str) -> float:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Missing OpenROAD metric for pattern: {pattern}")
        return float(match.group(1))

    violations = re.findall(r"Number of violations\s*=\s*(\d+)", stdout, re.IGNORECASE)
    if not violations:
        raise ValueError("Missing OpenROAD metric for DRC violations")
    total_wire_lengths = re.findall(r"Total wire length\s*=\s*([-\d.]+)\s*um\.?", stdout, re.IGNORECASE)
    routed_wirelength = float(total_wire_lengths[-1]) if total_wire_lengths else _sum_global_route_wirelength(stdout)

    return {
        "total_routed_wirelength": routed_wirelength,
        "tns": extract_float(r"tns\s+max\s+([-\d.]+)"),
        "wns": extract_float(r"wns\s+max\s+([-\d.]+)"),
        "drc_violations": int(violations[-1]),
    }


def build_openroad_script(
    lef_path: str,
    def_path: str,
    placement_mode: str,
    placement: dict[str, tuple[float, float]] | None = None,
    orientations: dict[str, str] | None = None,
    liberty_path: str | None = None,
    sdc_path: str | None = None,
    lef_paths: list[str] | tuple[str, ...] | None = None,
    detailed_route_end_iter: int = 0,
    global_placement_density: float | None = None,
    write_def_path: str | None = None,
) -> str:
    bridge = OpenROADBridge()
    return bridge.build_script(
        lef_paths or lef_path,
        def_path,
        placement or {},
        placement_mode=placement_mode,
        orientations=orientations,
        liberty_path=liberty_path,
        sdc_path=sdc_path,
        detailed_route_end_iter=detailed_route_end_iter,
        global_placement_density=global_placement_density,
        write_def_path=write_def_path,
    )


def evaluate_openroad_request(
    request: EvaluationRequest,
    bridge: OpenROADBridge | None = None,
    placement: dict[str, tuple[float, float]] | None = None,
    orientations: dict[str, str] | None = None,
) -> dict[str, float | int]:
    evaluator = bridge or OpenROADBridge(docker_image="openroad/orfs")
    if not evaluator.is_available():
        target = evaluator.docker_image or evaluator.binary_name
        raise RuntimeError(f"OpenROAD target '{target}' is not available")

    artifact_dir = Path(request.artifact_dir)
    script_path = artifact_dir / f".{request.placement_mode}_acceptance.tcl"
    script_text = build_openroad_script(
        lef_path=request.lef_path,
        def_path=request.def_path,
        placement_mode=request.placement_mode,
        placement=placement,
        orientations=orientations,
        liberty_path=request.liberty_path,
        sdc_path=request.sdc_path,
        lef_paths=request.lef_paths or (request.lef_path,),
        detailed_route_end_iter=request.detailed_route_end_iter,
        global_placement_density=request.placement_density,
        write_def_path=request.write_def_path,
    )
    script_path.write_text(script_text)

    host_lef_path = artifact_dir / (request.lef_paths[0] if request.lef_paths else request.lef_path)
    host_def_path = artifact_dir / request.def_path
    start = time.perf_counter()
    try:
        referenced_paths = [artifact_dir / path for path in (request.lef_paths or (request.lef_path,))]
        if request.liberty_path is not None:
            referenced_paths.append(artifact_dir / request.liberty_path)
        if request.sdc_path is not None:
            referenced_paths.append(artifact_dir / request.sdc_path)
        completed = evaluator.run_script(
            host_lef_path,
            host_def_path,
            script_path,
            referenced_paths=referenced_paths,
            working_directory=artifact_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        script_path.unlink(missing_ok=True)

    metrics = parse_openroad_metrics(completed.stdout)
    metrics["runtime_seconds"] = time.perf_counter() - start
    return metrics


def _sum_global_route_wirelength(stdout: str) -> float:
    if "Global route wire length by layer:" not in stdout:
        raise ValueError("Missing OpenROAD routed wirelength summary")
    total = 0.0
    for match in re.finditer(r"^\s*[A-Za-z0-9_]+\s+([-\d.]+)um", stdout, re.MULTILINE):
        total += float(match.group(1))
    return total
