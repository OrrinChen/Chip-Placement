from pathlib import Path

from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS
from chip.ispd_benchmarks import ISPDBenchmark
from chip.ispd_benchmarks import prepare_benchmark_artifacts
from chip.ispd_benchmarks import resolve_artifact_paths


def test_acceptance_benchmarks_cover_three_scale_buckets():
    assert len(ACCEPTANCE_BENCHMARKS) == 3
    assert [benchmark.scale_bucket for benchmark in ACCEPTANCE_BENCHMARKS] == ["small", "medium", "large"]


def test_acceptance_benchmarks_match_validated_phase1_set():
    assert [benchmark.name for benchmark in ACCEPTANCE_BENCHMARKS] == [
        "mgc_pci_bridge32_1",
        "mgc_pci_bridge32_2",
        "mgc_des_perf_2",
    ]
    assert all(benchmark.compatibility_status == "stable" for benchmark in ACCEPTANCE_BENCHMARKS)


def test_resolve_artifact_paths_points_to_benchmark_bundle(tmp_path):
    benchmark_root = tmp_path / "benchmarks"

    paths = resolve_artifact_paths(benchmark_root, "mgc_pci_bridge32_1")

    assert paths["tech_lef_path"] == benchmark_root / "mgc_pci_bridge32_1" / "tech.lef"
    assert paths["cells_lef_path"] == benchmark_root / "mgc_pci_bridge32_1" / "cells.lef"
    assert paths["def_path"] == benchmark_root / "mgc_pci_bridge32_1" / "floorplan.def"


def test_prepare_benchmark_artifacts_strips_vias_and_specialnets_and_writes_sdc(tmp_path):
    benchmark = ISPDBenchmark(
        name="demo",
        source_family="ISPD demo",
        scale_bucket="small",
        artifact_dir="demo",
        design_name="demo",
        verilog_filename="design.v",
        def_filename="floorplan.def",
        tech_lef_filename="tech.lef",
        cells_lef_filename="cells.lef",
        liberty_relpath="ispd2012/lib/contest.lib",
        clock_port="ispd_clk",
        clock_period_ps=660.0,
    )
    benchmark_root = tmp_path / "benchmarks"
    design_dir = benchmark_root / "demo"
    design_dir.mkdir(parents=True)
    (design_dir / "tech.lef").write_text("VERSION 5.7 ;\nSITE core\n  CLASS CORE ;\n  SIZE 0.2 BY 2.0 ;\nEND core\nEND LIBRARY\n")
    (design_dir / "cells.lef").write_text("VERSION 5.7 ;\nEND LIBRARY\n")
    (design_dir / "design.v").write_text("module demo(input ispd_clk, input a, output y); endmodule\n")
    (design_dir / "floorplan.def").write_text(
        "\n".join(
            [
                "VERSION 5.7 ;",
                "UNITS DISTANCE MICRONS 1000 ;",
                "DESIGN demo ;",
                "DIEAREA ( 0 0 ) ( 1000 1000 ) ;",
                "ROW core_SITE_ROW_0 core 0 0 N DO 5 BY 1 STEP 200 0 ;",
                "VIAS 1 ;",
                "  - VIA23_2cut_E",
                "    + RECT metal2 ( -50 -90 ) ( 250 90 ) ;",
                "END VIAS",
                "SPECIALNETS 1 ;",
                "  - vdd + ROUTED metal4 0 ( 0 0 ) ;",
                "END SPECIALNETS",
                "COMPONENTS 1 ;",
                "  - U1 INV_X1 + UNPLACED ;",
                "END COMPONENTS",
                "NETS 0 ;",
                "END NETS",
                "END DESIGN",
            ]
        )
    )
    (tmp_path / "ispd2012" / "lib").mkdir(parents=True)
    (tmp_path / "ispd2012" / "lib" / "contest.lib").write_text("library(contest) {}\n")

    prepared = prepare_benchmark_artifacts(benchmark_root=benchmark_root, benchmark=benchmark)
    clean_def_text = Path(prepared["def_path"]).read_text()
    sdc_text = Path(prepared["sdc_path"]).read_text()

    assert "VIAS" not in clean_def_text
    assert "SPECIALNETS" not in clean_def_text
    assert "create_clock -name mclk -period 660.0 [get_ports ispd_clk]" in sdc_text
    assert Path(prepared["site_fix_lef_path"]).exists()
