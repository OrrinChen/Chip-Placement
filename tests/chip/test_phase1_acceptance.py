from chip.experiments.phase1_acceptance import _select_injectable_instances
from chip.experiments.phase1_acceptance import summarize_acceptance_row
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from chip.problem import Net


def test_summarize_acceptance_row_combines_baseline_and_openroad_metrics():
    row = summarize_acceptance_row(
        benchmark_name="ibm01",
        scale_bucket="small",
        design_stats={"node_count": 1000, "net_count": 1200},
        sa_summary={"hpwl": 10.0, "is_valid": True, "runtime_seconds": 1.5},
        built_in_metrics={
            "total_routed_wirelength": 100.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 5.0,
        },
        sa_metrics={
            "total_routed_wirelength": 110.0,
            "wns": -0.2,
            "tns": -2.0,
            "drc_violations": 1,
            "runtime_seconds": 2.5,
        },
    )

    assert row["benchmark"] == "ibm01"
    assert row["scale_bucket"] == "small"
    assert row["sa_hpwl"] == 10.0
    assert row["built_in_total_routed_wirelength"] == 100.0
    assert row["sa_drc_violations"] == 1
    assert row["phase1_gate_status"] == "infra_pass / quality_pass"
    assert row["phase1_gate_reason"] is None


def test_summarize_acceptance_row_marks_quality_pending_when_sa_is_invalid():
    row = summarize_acceptance_row(
        benchmark_name="ibm01",
        scale_bucket="small",
        design_stats={"node_count": 1000, "net_count": 1200},
        sa_summary={"hpwl": 10.0, "is_valid": False, "runtime_seconds": 1.5},
        built_in_metrics={
            "total_routed_wirelength": 100.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 5.0,
        },
        sa_metrics={
            "total_routed_wirelength": 110.0,
            "wns": -0.2,
            "tns": -2.0,
            "drc_violations": 1,
            "runtime_seconds": 2.5,
        },
    )

    assert row["phase1_gate_status"] == "infra_pass / quality_pending"
    assert "not legal yet" in row["phase1_gate_reason"]


def test_run_acceptance_suite_surfaces_benchmark_failures(monkeypatch, tmp_path):
    from chip.experiments import phase1_acceptance as module

    benchmark_name = module.ACCEPTANCE_BENCHMARKS[0].name
    monkeypatch.setattr(
        module,
        "run_single_benchmark_acceptance",
        lambda *args, **kwargs: {"benchmark": benchmark_name, "error": "routing_failed"},
    )

    rows = module.run_acceptance_suite(
        benchmark_root=tmp_path,
        output_path=tmp_path / "results.json",
        benchmark_names=[benchmark_name],
    )

    assert rows == [{"benchmark": benchmark_name, "error": "routing_failed"}]


def test_select_injectable_instances_excludes_fixed_pins_and_keeps_movable_macros():
    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 100.0, 100.0),
        macros={
            "U1": Macro(name="U1", kind="M1", width=1.0, height=1.0, x=1.0, y=2.0, orientation="N", is_fixed=False),
            "PIN_A": Macro(name="PIN_A", kind="PIN", width=0.0, height=0.0, x=5.0, y=6.0, orientation="N", is_fixed=True),
            "U_FIXED": Macro(
                name="U_FIXED",
                kind="M2",
                width=1.0,
                height=1.0,
                x=3.0,
                y=4.0,
                orientation="FS",
                is_fixed=True,
            ),
        },
        nets=[Net(name="N1", pins=("U1", "PIN_A"))],
    )

    placement, orientations = _select_injectable_instances(
        problem,
        {"U1": (10.13, 20.0), "PIN_A": (30.0, 40.0), "U_FIXED": (50.0, 60.0)},
    )

    assert placement == {"U1": (10.2, 2.0)}
    assert orientations == {"U1": "N"}


def test_run_single_benchmark_acceptance_uses_built_in_seed_def_for_sa(monkeypatch, tmp_path):
    from chip.experiments import phase1_acceptance as module

    artifact_dir = tmp_path / "mgc_pci_bridge32_1"
    artifact_dir.mkdir()
    seed_def_path = artifact_dir / "seed_placed.def"
    seed_def_path.write_text("VERSION 5.8 ;\nEND DESIGN\n")
    benchmark = module.ACCEPTANCE_BENCHMARKS[0]
    load_calls: list[str] = []
    request_calls: list[dict[str, object]] = []
    eval_calls: list[dict[str, object]] = []

    prepared = {
        "artifact_dir": artifact_dir,
        "tech_lef_path": artifact_dir / "tech.lef",
        "site_fix_lef_path": artifact_dir / "site_fix.lef",
        "cells_lef_path": artifact_dir / "cells.lef",
        "def_path": artifact_dir / "floorplan_clean.def",
        "liberty_path": artifact_dir / "contest.lib",
        "sdc_path": artifact_dir / "acceptance.sdc",
    }
    for path in prepared.values():
        if hasattr(path, "write_text") and path != artifact_dir:
            path.write_text("")

    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 100.0, 100.0),
        macros={"U1": Macro(name="U1", kind="M1", width=1.0, height=1.0, x=1.0, y=2.0, orientation="N", is_fixed=False)},
        nets=[Net(name="N1", pins=("U1",))],
    )

    class FakeGraph:
        node_ids = ("U1",)

    monkeypatch.setattr(module, "prepare_benchmark_artifacts", lambda **_: prepared)

    def fake_load_chip_problem(_lef_path, def_path):
        load_calls.append(str(def_path))
        return FakeGraph(), problem

    monkeypatch.setattr(module, "load_chip_problem", fake_load_chip_problem)
    monkeypatch.setattr(module, "_run_sa_baseline", lambda *_args, **_kwargs: ({"U1": (10.13, 20.0)}, {"hpwl": 1.0, "is_valid": True, "runtime_seconds": 0.1}))

    def fake_build_evaluation_request(**kwargs):
        request_calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(module, "build_evaluation_request", fake_build_evaluation_request)
    monkeypatch.setattr(module, "OpenROADBridge", lambda **_: object())

    def fake_evaluate_openroad_request(request, **kwargs):
        eval_calls.append({"request": request, **kwargs})
        return {
            "total_routed_wirelength": 10.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 1.0,
        }

    monkeypatch.setattr(module, "evaluate_openroad_request", fake_evaluate_openroad_request)

    row = module.run_single_benchmark_acceptance(tmp_path, benchmark)

    assert row["benchmark"] == benchmark.name
    assert str(prepared["def_path"]) in load_calls[0]
    assert str(seed_def_path) in load_calls[1]
    assert request_calls[0]["write_def_path"] == "seed_placed.def"
    assert request_calls[1]["def_path"] == "seed_placed.def"
    assert eval_calls[1]["placement"] == {"U1": (10.2, 2.0)}
