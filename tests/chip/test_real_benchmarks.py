from chip.experiments.real_benchmarks import summarize_design_result


def test_summarize_design_result_extracts_phase1_metrics():
    result = {
        "placements": {
            "initial": {"hpwl": 10.0, "is_valid": True},
            "analytical": {"status": "skipped", "reason": "node_limit_exceeded"},
            "sa": {"hpwl": 8.0, "is_valid": False},
        },
        "openroad_available": True,
    }
    design_stats = {"node_count": 100, "fixed_node_count": 12, "net_count": 50}

    summary = summarize_design_result("gcd", design_stats, result)

    assert summary["design"] == "gcd"
    assert summary["initial_hpwl"] == 10.0
    assert summary["sa_hpwl"] == 8.0
    assert summary["analytical_status"] == "skipped"
    assert summary["openroad_available"] is True


def test_summarize_many_designs_keeps_missing_design_errors_visible():
    from chip.experiments.real_benchmarks import summarize_many_designs

    rows = summarize_many_designs(
        {
            "gcd": {"summary": {"design": "gcd"}, "error": None},
            "aes": {"summary": None, "error": "missing_odb"},
        }
    )

    assert rows[0]["design"] == "gcd"
    assert rows[1]["design"] == "aes"
    assert rows[1]["error"] == "missing_odb"


def test_design_artifact_paths_points_to_exported_bundle(tmp_path):
    from chip.experiments.real_benchmarks import design_artifact_paths

    export_root = tmp_path / "orfs-export"

    paths = design_artifact_paths(export_root, "gcd")

    assert paths["lef_path"] == export_root / "nangate45" / "gcd" / "NangateOpenCellLibrary.tech.lef"
    assert paths["def_path"] == export_root / "nangate45" / "gcd" / "3_place.def"


def test_run_real_benchmark_suite_summarizes_ready_design(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from chip.experiments import real_benchmarks as module

    class FakeExporter:
        def __init__(self, *args, **kwargs):
            pass

        def export_design(self, design_name):
            return {
                "status": "ready",
                "bundle": SimpleNamespace(
                    def_path=tmp_path / "orfs-export" / "nangate45" / design_name / "3_place.def",
                    tech_lef_path=tmp_path / "orfs-export" / "nangate45" / design_name / "NangateOpenCellLibrary.tech.lef",
                ),
            }

    monkeypatch.setattr(module, "ORFSDesignExporter", FakeExporter)
    monkeypatch.setattr(
        module,
        "load_chip_problem",
        lambda lef_path, def_path: (
            SimpleNamespace(node_ids=("a", "b", "c"), global_features={"fixed_nodes": {"a"}}),
            SimpleNamespace(nets=["n1", "n2"]),
        ),
    )
    monkeypatch.setattr(
        module,
        "run_baseline_experiment",
        lambda *args, **kwargs: {
            "placements": {
                "initial": {"hpwl": 10.0, "is_valid": True},
                "analytical": {"status": "skipped", "reason": "node_limit_exceeded"},
                "sa": {"hpwl": 8.5, "is_valid": True},
            },
            "openroad_available": True,
        },
    )

    rows = module.run_real_benchmark_suite(
        work_root=tmp_path / "orfs-work",
        export_root=tmp_path / "orfs-export",
        design_names=["gcd"],
    )

    assert rows == [
        {
            "design": "gcd",
            "node_count": 3,
            "fixed_node_count": 1,
            "net_count": 2,
            "initial_hpwl": 10.0,
            "initial_is_valid": True,
            "sa_hpwl": 8.5,
            "sa_is_valid": True,
            "analytical_status": "skipped",
            "openroad_available": True,
        }
    ]


def test_run_real_benchmark_suite_surfaces_export_errors(monkeypatch, tmp_path):
    from chip.experiments import real_benchmarks as module

    class FakeExporter:
        def __init__(self, *args, **kwargs):
            pass

        def export_design(self, design_name):
            return {"status": "missing_odb", "bundle": None}

    monkeypatch.setattr(module, "ORFSDesignExporter", FakeExporter)

    rows = module.run_real_benchmark_suite(
        work_root=tmp_path / "orfs-work",
        export_root=tmp_path / "orfs-export",
        design_names=["ibex"],
    )

    assert rows == [{"design": "ibex", "error": "missing_odb"}]
