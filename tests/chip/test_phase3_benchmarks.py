from __future__ import annotations

from chip.experiments.phase3_benchmarks import build_phase3_benchmark_ladder


def test_phase3_benchmark_ladder_has_tiny_small_medium():
    ladder = build_phase3_benchmark_ladder()

    assert tuple(item["name"] for item in ladder) == ("tiny", "small", "medium")


def test_phase3_benchmark_ladder_increases_movable_macro_count():
    ladder = build_phase3_benchmark_ladder()
    movable_counts = [item["movable_macro_count"] for item in ladder]

    assert movable_counts[0] < movable_counts[1] < movable_counts[2]
