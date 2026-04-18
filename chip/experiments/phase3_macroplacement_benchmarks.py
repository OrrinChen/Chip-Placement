from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MacroPlacementBenchmarkSpec:
    name: str
    variant: str
    platform: str
    repo_root: Path
    testcase_dir: Path
    flow_dir: Path
    enablement_dir: Path
    reference_def_path: Path
    floorplan_def_path: Path
    manual_def_path: Path | None
    grouped_netlist_pbtxt_path: Path | None
    tech_lef_path: Path
    macro_lef_path: Path
    lef_paths: tuple[Path, ...]
    standard_liberty_path: Path
    macro_liberty_paths: tuple[Path, ...]
    sdc_path: Path


def default_macroplacement_repo_root(start: Path | None = None) -> Path:
    search_root = (start or Path(__file__)).resolve()
    for candidate in [search_root, *search_root.parents]:
        repo_root = candidate / "tmp" / "MacroPlacement"
        if repo_root.exists():
            return repo_root
    raise FileNotFoundError("Could not locate tmp/MacroPlacement in current worktree")


def resolve_macroplacement_benchmark(
    *,
    repo_root: Path | None = None,
    variant: str,
    platform: str = "NanGate45",
) -> MacroPlacementBenchmarkSpec:
    resolved_repo_root = (repo_root or default_macroplacement_repo_root()).resolve()
    if not resolved_repo_root.exists():
        raise FileNotFoundError(f"MacroPlacement repo root does not exist: {resolved_repo_root}")

    testcase_dir = resolved_repo_root / "Testcases" / variant
    flow_dir = resolved_repo_root / "Flows" / platform / variant
    enablement_dir = resolved_repo_root / "Enablements" / platform
    if not flow_dir.exists():
        raise FileNotFoundError(f"Missing flow directory: {flow_dir}")
    if not enablement_dir.exists():
        raise FileNotFoundError(f"Missing enablement directory: {enablement_dir}")
    if not testcase_dir.exists():
        raise FileNotFoundError(f"Missing testcase directory: {testcase_dir}")

    def_dir = flow_dir / "def"
    reference_defs = sorted(def_dir.glob("*_fp_placed_macros.def"))
    if len(reference_defs) != 1:
        raise FileNotFoundError(f"Expected exactly one *_fp_placed_macros.def under {def_dir}, found {len(reference_defs)}")
    reference_def_path = reference_defs[0]
    floorplan_def_path = def_dir / reference_def_path.name.replace("_placed_macros", "")
    if not floorplan_def_path.exists():
        floorplan_defs = sorted(path for path in def_dir.glob("*_fp.def") if path != reference_def_path)
        if len(floorplan_defs) != 1:
            raise FileNotFoundError(f"Expected exactly one matching floorplan DEF under {def_dir}, found {len(floorplan_defs)}")
        floorplan_def_path = floorplan_defs[0]
    manual_def_path = (def_dir / "manual_floorplan.def") if (def_dir / "manual_floorplan.def").exists() else None
    if manual_def_path is None:
        manual_candidates = sorted((def_dir / "manual_macro_placement").glob("*.def")) if (def_dir / "manual_macro_placement").exists() else []
        manual_def_path = manual_candidates[0] if manual_candidates else None
    grouped_netlist_candidate = flow_dir / "netlist" / "output_CT_Grouping" / "netlist.pb.txt"
    grouped_netlist_pbtxt_path = grouped_netlist_candidate if grouped_netlist_candidate.exists() else None
    tech_lef_path = enablement_dir / "lef" / "NangateOpenCellLibrary.tech.lef"
    macro_lef_path = enablement_dir / "lef" / "NangateOpenCellLibrary.macro.mod.lef"
    all_lef_paths = _ordered_lef_paths(enablement_dir / "lef", tech_lef_path)
    standard_liberty_path = enablement_dir / "lib" / "NangateOpenCellLibrary_typical.lib"
    macro_liberty_paths = tuple(sorted(path for path in (enablement_dir / "lib").glob("fakeram45_*.lib")))
    sdc_paths = sorted((flow_dir / "constraints").glob("*.sdc"))
    if len(sdc_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one .sdc under {flow_dir / 'constraints'}, found {len(sdc_paths)}")
    sdc_path = sdc_paths[0]

    required_paths = [
        reference_def_path,
        floorplan_def_path,
        tech_lef_path,
        macro_lef_path,
        standard_liberty_path,
        sdc_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"MacroPlacement benchmark assets missing: {missing}")

    name = f"{variant}_{platform.lower().replace('nangate45', 'ng45')}"
    return MacroPlacementBenchmarkSpec(
        name=name,
        variant=variant,
        platform=platform,
        repo_root=resolved_repo_root,
        testcase_dir=testcase_dir,
        flow_dir=flow_dir,
        enablement_dir=enablement_dir,
        reference_def_path=reference_def_path,
        floorplan_def_path=floorplan_def_path,
        manual_def_path=manual_def_path,
        grouped_netlist_pbtxt_path=grouped_netlist_pbtxt_path,
        tech_lef_path=tech_lef_path,
        macro_lef_path=macro_lef_path,
        lef_paths=all_lef_paths,
        standard_liberty_path=standard_liberty_path,
        macro_liberty_paths=macro_liberty_paths,
        sdc_path=sdc_path,
    )


def resolve_ariane_benchmark(
    *,
    repo_root: Path | None = None,
    variant: str = "ariane133",
    platform: str = "NanGate45",
) -> MacroPlacementBenchmarkSpec:
    return resolve_macroplacement_benchmark(
        repo_root=repo_root,
        variant=variant,
        platform=platform,
    )


def materialize_combined_lef(spec: MacroPlacementBenchmarkSpec, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{spec.name}.combined.lef"
    text = "\n".join(path.read_text() for path in spec.lef_paths)
    combined_path.write_text(text)
    return combined_path


def materialize_combined_liberty(spec: MacroPlacementBenchmarkSpec, *, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{spec.name}.combined.lib"
    library_paths = (spec.standard_liberty_path, *spec.macro_liberty_paths)
    text = "\n".join(path.read_text() for path in library_paths)
    combined_path.write_text(text)
    return combined_path


def _ordered_lef_paths(lef_dir: Path, tech_lef_path: Path) -> tuple[Path, ...]:
    other_paths = sorted(path for path in lef_dir.glob("*.lef") if path != tech_lef_path)
    return (tech_lef_path, *other_paths)
