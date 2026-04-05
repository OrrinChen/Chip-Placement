from __future__ import annotations

import re
from dataclasses import dataclass

from chip.openroad_bridge import OpenROADBridge


@dataclass(frozen=True)
class EvaluationRequest:
    benchmark_name: str
    lef_path: str
    def_path: str
    placement_mode: str


def build_evaluation_request(
    benchmark_name: str,
    lef_path: str,
    def_path: str,
    placement_mode: str,
) -> EvaluationRequest:
    return EvaluationRequest(
        benchmark_name=benchmark_name,
        lef_path=lef_path,
        def_path=def_path,
        placement_mode=placement_mode,
    )


def parse_openroad_metrics(stdout: str) -> dict[str, float | int]:
    def extract_float(pattern: str) -> float:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Missing OpenROAD metric for pattern: {pattern}")
        return float(match.group(1))

    def extract_int(pattern: str) -> int:
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match is None:
            raise ValueError(f"Missing OpenROAD metric for pattern: {pattern}")
        return int(match.group(1))

    return {
        "total_routed_wirelength": extract_float(r"total routed wire length\s*=\s*([-\d.]+)"),
        "tns": extract_float(r"report_tns\s+([-\d.]+)"),
        "wns": extract_float(r"report_wns\s+([-\d.]+)"),
        "drc_violations": extract_int(r"drc violation count\s*=\s*(\d+)"),
    }


def build_openroad_script(
    lef_path: str,
    def_path: str,
    placement_mode: str,
    placement: dict[str, tuple[float, float]] | None = None,
    orientations: dict[str, str] | None = None,
) -> str:
    bridge = OpenROADBridge()
    return bridge.build_script(
        lef_path,
        def_path,
        placement or {},
        placement_mode=placement_mode,
        orientations=orientations,
    )
