from pathlib import Path

from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS
from chip.ispd_benchmarks import resolve_artifact_paths


def test_acceptance_benchmarks_cover_three_scale_buckets():
    assert len(ACCEPTANCE_BENCHMARKS) == 3
    assert [benchmark.scale_bucket for benchmark in ACCEPTANCE_BENCHMARKS] == ["small", "medium", "large"]


def test_resolve_artifact_paths_points_to_benchmark_bundle(tmp_path):
    benchmark_root = tmp_path / "benchmarks"

    paths = resolve_artifact_paths(benchmark_root, "ibm01")

    assert paths["lef_path"] == benchmark_root / "ibm01" / "design.lef"
    assert paths["def_path"] == benchmark_root / "ibm01" / "design.def"
