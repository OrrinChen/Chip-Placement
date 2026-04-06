from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class ISPDBenchmark:
    name: str
    source_family: str
    scale_bucket: str
    artifact_dir: str
    design_name: str
    verilog_filename: str
    def_filename: str
    tech_lef_filename: str
    cells_lef_filename: str
    liberty_relpath: str
    clock_port: str
    clock_period_ps: float
    official_sdc_relpath: str | None = None
    placement_density: float | None = None
    sa_iterations: int = 1
    detailed_route_end_iter: int = 0
    compatibility_status: str = "candidate"


ACCEPTANCE_BENCHMARKS = [
    ISPDBenchmark(
        name="mgc_pci_bridge32_1",
        source_family="ISPD 2014 Benchmark A",
        scale_bucket="small",
        artifact_dir="mgc_pci_bridge32_1",
        design_name="pci_bridge32",
        verilog_filename="design.v",
        def_filename="floorplan.def",
        tech_lef_filename="tech.lef",
        cells_lef_filename="cells.lef",
        liberty_relpath="ispd2012/lib/contest.lib",
        official_sdc_relpath="ispd2012/pci_bridge32_fast/pci_bridge32_fast.sdc",
        clock_port="ispd_clk",
        clock_period_ps=660.0,
        placement_density=0.86,
        compatibility_status="stable",
    ),
    ISPDBenchmark(
        name="mgc_pci_bridge32_2",
        source_family="ISPD 2014 Benchmark A",
        scale_bucket="medium",
        artifact_dir="mgc_pci_bridge32_2",
        design_name="pci_bridge32",
        verilog_filename="design.v",
        def_filename="floorplan.def",
        tech_lef_filename="tech.lef",
        cells_lef_filename="cells.lef",
        liberty_relpath="ispd2012/lib/contest.lib",
        official_sdc_relpath="ispd2012/pci_bridge32_fast/pci_bridge32_fast.sdc",
        clock_port="ispd_clk",
        clock_period_ps=660.0,
        placement_density=0.86,
        compatibility_status="stable",
    ),
    ISPDBenchmark(
        name="mgc_des_perf_2",
        source_family="ISPD 2014 Benchmark A",
        scale_bucket="large",
        artifact_dir="mgc_des_perf_2",
        design_name="des_perf",
        verilog_filename="design.v",
        def_filename="floorplan.def",
        tech_lef_filename="tech.lef",
        cells_lef_filename="cells.lef",
        liberty_relpath="ispd2012/lib/contest.lib",
        official_sdc_relpath="ispd2012/des_perf_fast/des_perf_fast.sdc",
        clock_port="ispd_clk",
        clock_period_ps=660.0,
        placement_density=0.88,
        compatibility_status="stable",
    ),
]


def resolve_artifact_paths(benchmark_root: Path, benchmark_name: str) -> dict[str, Path]:
    benchmark = next((item for item in ACCEPTANCE_BENCHMARKS if item.name == benchmark_name), None)
    root = Path(benchmark_root) / benchmark_name
    shared_root = _resolve_shared_root(benchmark_root)
    return {
        "tech_lef_path": root / (benchmark.tech_lef_filename if benchmark else "tech.lef"),
        "cells_lef_path": root / (benchmark.cells_lef_filename if benchmark else "cells.lef"),
        "def_path": root / (benchmark.def_filename if benchmark else "floorplan.def"),
        "verilog_path": root / (benchmark.verilog_filename if benchmark else "design.v"),
        "liberty_path": shared_root / (benchmark.liberty_relpath if benchmark else "ispd2012/lib/contest.lib"),
        "official_sdc_path": shared_root / benchmark.official_sdc_relpath
        if benchmark and benchmark.official_sdc_relpath
        else root / "acceptance.sdc",
    }


def prepare_benchmark_artifacts(benchmark_root: Path, benchmark: ISPDBenchmark) -> dict[str, Path]:
    source_paths = resolve_artifact_paths(benchmark_root, benchmark.name)
    artifact_dir = Path(benchmark_root) / benchmark.artifact_dir
    clean_def_path = artifact_dir / f"{Path(benchmark.def_filename).stem}_clean.def"
    site_fix_lef_path = artifact_dir / "site_fix.lef"
    sdc_path = artifact_dir / "acceptance.sdc"

    _write_clean_def(source_paths["def_path"], clean_def_path)
    _write_site_fix(site_fix_lef_path)
    if benchmark.official_sdc_relpath and source_paths["official_sdc_path"].exists():
        shutil.copyfile(source_paths["official_sdc_path"], sdc_path)
    else:
        _write_synthetic_sdc(sdc_path, benchmark)

    return {
        "artifact_dir": artifact_dir,
        "tech_lef_path": source_paths["tech_lef_path"],
        "site_fix_lef_path": site_fix_lef_path,
        "cells_lef_path": source_paths["cells_lef_path"],
        "def_path": clean_def_path,
        "verilog_path": source_paths["verilog_path"],
        "liberty_path": source_paths["liberty_path"],
        "sdc_path": sdc_path,
    }


def _resolve_shared_root(benchmark_root: Path) -> Path:
    root = Path(benchmark_root)
    if (root / "ispd2012").exists():
        return root
    if (root.parent / "ispd2012").exists():
        return root.parent
    return root


def _write_clean_def(source_path: Path, destination_path: Path) -> None:
    drop_section = False
    filtered_lines: list[str] = []
    for line in Path(source_path).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("VIAS") or stripped.startswith("SPECIALNETS"):
            drop_section = True
            continue
        if stripped.startswith("END VIAS") or stripped.startswith("END SPECIALNETS"):
            drop_section = False
            continue
        if not drop_section:
            filtered_lines.append(line)
    destination_path.write_text("\n".join(filtered_lines) + "\n")


def _write_site_fix(destination_path: Path) -> None:
    destination_path.write_text(
        "\n".join(
            [
                "VERSION 5.7 ;",
                "SITE core",
                "  CLASS CORE ;",
                "  SIZE 0.200 BY 2.000 ;",
                "END core",
                "END LIBRARY",
                "",
            ]
        )
    )


def _write_synthetic_sdc(destination_path: Path, benchmark: ISPDBenchmark) -> None:
    destination_path.write_text(
        "\n".join(
            [
                f"create_clock -name mclk -period {benchmark.clock_period_ps:.1f} [get_ports {benchmark.clock_port}]",
                "set_input_delay 0.0 [all_inputs] -clock mclk",
                "set_output_delay 0.0 [all_outputs] -clock mclk",
                "",
            ]
        )
    )
