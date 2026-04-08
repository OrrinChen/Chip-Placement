from __future__ import annotations

from pathlib import Path

from chip.netlist_parser import load_chip_problem
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net
from chip.problem import PlacementRow


def build_phase3_benchmark_ladder() -> list[dict[str, object]]:
    sample_paths = sample_artifact_paths()
    ladder = [
        {
            "name": "tiny",
            "problem": build_tiny_problem(),
            "grid_shape": (4, 4),
        },
        {
            "name": "small",
            "problem": load_sample_problem(),
            "grid_shape": (6, 6),
            "artifact_paths": sample_paths,
        },
        {
            "name": "medium",
            "problem": build_medium_problem(),
            "grid_shape": (8, 8),
        },
    ]
    for benchmark in ladder:
        problem: ChipProblem = benchmark["problem"]
        benchmark["movable_macro_count"] = sum(
            1 for macro in problem.macros.values() if not macro.is_fixed and macro.kind != "PIN"
        )
        benchmark["node_count"] = len(problem.macros)
        benchmark["net_count"] = len(problem.nets)
    return ladder


def build_tiny_problem() -> ChipProblem:
    return ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=12.0, max_y=4.0),
        macros={
            "u1": Macro(name="u1", kind="CELL", width=1.0, height=1.0, x=0.0, y=0.0, is_fixed=False),
            "u2": Macro(name="u2", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0, is_fixed=True),
        },
        nets=[Net(name="n1", pins=("u1", "u2"))],
        rows=(
            PlacementRow(name="ROW_0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=12, step_x=1.0, step_y=0.0),
            PlacementRow(name="ROW_1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=12, step_x=1.0, step_y=0.0),
        ),
    )


def load_sample_problem() -> ChipProblem:
    paths = sample_artifact_paths()
    _, problem = load_chip_problem(paths["lef_path"], paths["def_path"])
    return problem


def sample_artifact_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    lef_path = root / "chip" / "benchmarks" / "sample.lef"
    def_path = root / "chip" / "benchmarks" / "sample.def"
    return {
        "lef_path": lef_path,
        "def_path": def_path,
    }


def build_medium_problem() -> ChipProblem:
    macros = {
        "clk": Macro(name="clk", kind="PIN", width=0.0, height=0.0, x=1.0, y=7.0, is_fixed=True),
        "out": Macro(name="out", kind="PIN", width=0.0, height=0.0, x=19.0, y=1.0, is_fixed=True),
        "m1": Macro(name="m1", kind="CELL", width=1.0, height=1.0, x=0.0, y=0.0),
        "m2": Macro(name="m2", kind="CELL", width=1.0, height=1.0, x=2.0, y=0.0),
        "m3": Macro(name="m3", kind="CELL", width=1.0, height=1.0, x=4.0, y=0.0),
        "m4": Macro(name="m4", kind="CELL", width=1.0, height=1.0, x=6.0, y=0.0),
        "m5": Macro(name="m5", kind="CELL", width=1.0, height=1.0, x=8.0, y=0.0),
        "m6": Macro(name="m6", kind="CELL", width=1.0, height=1.0, x=10.0, y=0.0),
        "m7": Macro(name="m7", kind="CELL", width=1.0, height=1.0, x=12.0, y=0.0),
        "m8": Macro(name="m8", kind="CELL", width=1.0, height=1.0, x=14.0, y=0.0),
    }
    nets = [
        Net(name="n_clk_1", pins=("clk", "m1", "m2")),
        Net(name="n_clk_2", pins=("clk", "m3", "m4")),
        Net(name="n_core_1", pins=("m1", "m3", "m5")),
        Net(name="n_core_2", pins=("m2", "m4", "m6")),
        Net(name="n_core_3", pins=("m5", "m7", "out")),
        Net(name="n_core_4", pins=("m6", "m8", "out")),
        Net(name="n_mesh_1", pins=("m1", "m4", "m7")),
        Net(name="n_mesh_2", pins=("m2", "m5", "m8")),
    ]
    rows = tuple(
        PlacementRow(
            name=f"ROW_{index}",
            origin_x=0.0,
            origin_y=float(index * 2),
            orientation="N" if index % 2 == 0 else "FS",
            site_count=20,
            step_x=1.0,
            step_y=0.0,
        )
        for index in range(4)
    )
    return ChipProblem(
        die_area=DieArea(min_x=0.0, min_y=0.0, max_x=20.0, max_y=8.0),
        macros=macros,
        nets=nets,
        rows=rows,
    )
