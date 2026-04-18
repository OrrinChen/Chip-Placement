from __future__ import annotations

from pathlib import Path

import pytest

from chip.experiments.phase3_macroplacement_benchmarks import (
    MacroPlacementBenchmarkSpec,
    materialize_combined_lef,
    resolve_macroplacement_benchmark,
    resolve_ariane_benchmark,
)


def macroplacement_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tmp" / "MacroPlacement"


def test_resolve_ariane_benchmark_discovers_nangate45_assets() -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")

    assert isinstance(spec, MacroPlacementBenchmarkSpec)
    assert spec.name == "ariane133_ng45"
    assert spec.variant == "ariane133"
    assert spec.platform == "NanGate45"
    assert spec.reference_def_path.name == "ariane133_fp_placed_macros.def"
    assert spec.floorplan_def_path.name == "ariane133_fp.def"
    assert spec.manual_def_path is not None
    assert spec.manual_def_path.name == "manual_floorplan.def"
    assert spec.sdc_path.name == "ariane.sdc"
    assert spec.tech_lef_path.name == "NangateOpenCellLibrary.tech.lef"
    assert spec.macro_lef_path.name == "NangateOpenCellLibrary.macro.mod.lef"
    assert spec.standard_liberty_path.name == "NangateOpenCellLibrary_typical.lib"
    assert spec.flow_dir.exists()
    assert spec.testcase_dir.exists()
    assert all(path.exists() for path in spec.lef_paths)
    assert len(spec.macro_liberty_paths) >= 1


def test_materialize_combined_lef_contains_tech_and_macro_defs(tmp_path: Path) -> None:
    spec = resolve_ariane_benchmark(repo_root=macroplacement_repo_root(), variant="ariane133", platform="NanGate45")

    combined_path = materialize_combined_lef(spec, output_dir=tmp_path)

    text = combined_path.read_text()
    assert "MACRO fakeram45_256x16" in text
    assert "LAYER metal1" in text


def test_resolve_ariane_benchmark_rejects_missing_repo_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_ariane_benchmark(repo_root=tmp_path / "missing", variant="ariane133", platform="NanGate45")


def test_resolve_macroplacement_benchmark_supports_bp_quad() -> None:
    spec = resolve_macroplacement_benchmark(repo_root=macroplacement_repo_root(), variant="bp_quad", platform="NanGate45")

    assert isinstance(spec, MacroPlacementBenchmarkSpec)
    assert spec.name == "bp_quad_ng45"
    assert spec.variant == "bp_quad"
    assert spec.reference_def_path.name == "bsg_chip_fp_placed_macros.def"
    assert spec.floorplan_def_path.name == "bsg_chip_fp.def"
    assert spec.sdc_path.name == "bsg_chip.sdc"


def test_resolve_macroplacement_benchmark_supports_nvdla() -> None:
    spec = resolve_macroplacement_benchmark(repo_root=macroplacement_repo_root(), variant="nvdla", platform="NanGate45")

    assert isinstance(spec, MacroPlacementBenchmarkSpec)
    assert spec.name == "nvdla_ng45"
    assert spec.variant == "nvdla"
    assert spec.reference_def_path.name == "NV_NVDLA_partition_c_fp_placed_macros.def"
    assert spec.floorplan_def_path.name == "NV_NVDLA_partition_c_fp.def"
    assert spec.sdc_path.name == "NV_NVDLA_partition_c.sdc"
