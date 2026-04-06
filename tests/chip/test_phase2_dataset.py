import json

from chip.experiments.phase2_dataset import (
    _build_sample_placement,
    _sample_macro_perturbation,
    build_sampling_plan,
    collect_ground_truth,
    collect_sample_result,
    summarize_displacement_distribution,
    summarize_dataset_row,
)


def test_build_sampling_plan_expands_random_and_sa_samples():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="smoke",
        random_samples=2,
        sa_truncation_steps=(1, 3),
    )

    assert [item["source"] for item in plan] == [
        "random_perturbation",
        "random_perturbation",
        "sa_truncated",
        "sa_truncated",
    ]
    assert [item["sample_id"] for item in plan] == [
        "rand_0000",
        "rand_0001",
        "sa_0001",
        "sa_0003",
    ]
    assert {item["batch"] for item in plan} == {"smoke"}
    assert all(item["move_probability"] == 0.02 for item in plan if item["source"] == "random_perturbation")
    assert all(item["max_x_fraction"] == 0.05 for item in plan if item["source"] == "random_perturbation")
    assert all(item["row_window"] == 1 for item in plan if item["source"] == "random_perturbation")


def test_build_sampling_plan_can_skip_sa_samples():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="smoke",
        random_samples=2,
        sa_truncation_steps=(),
    )

    assert [item["source"] for item in plan] == [
        "random_perturbation",
        "random_perturbation",
    ]


def test_build_sampling_plan_propagates_random_controls():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="pilot",
        random_samples=1,
        sa_truncation_steps=(),
        random_move_probability=0.03,
        random_max_x_fraction=0.1,
        random_row_window=2,
    )

    assert plan == [
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0000",
            "source": "random_perturbation",
            "batch": "pilot",
            "random_steps": 5,
            "move_probability": 0.03,
            "max_x_fraction": 0.1,
            "row_window": 2,
        }
    ]


def test_build_sampling_plan_uses_layered_random_defaults_for_batch_generation():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="full",
        random_samples=10,
        sa_truncation_steps=(),
    )

    random_rows = [row for row in plan if row["source"] == "random_perturbation"]
    assert len(random_rows) == 10
    assert sum(1 for row in random_rows if row["move_probability"] == 0.1) == 7
    assert sum(1 for row in random_rows if row["move_probability"] == 0.2) == 3
    assert {row["max_x_fraction"] for row in random_rows} == {0.1}
    assert {row["row_window"] for row in random_rows} == {1}


def test_build_sampling_plan_can_override_layered_defaults_with_single_tier():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="pilot",
        random_samples=4,
        sa_truncation_steps=(),
        random_tiers=(
            {
                "move_probability": 0.05,
                "max_x_fraction": 0.1,
                "row_window": 1,
                "weight": 1.0,
            },
        ),
    )

    random_rows = [row for row in plan if row["source"] == "random_perturbation"]
    assert len(random_rows) == 4
    assert {row["move_probability"] for row in random_rows} == {0.05}


def test_build_sample_placement_random_perturbation_moves_many_cells():
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net
    from chip.problem import PlacementRow

    macros = {
        f"U{index}": Macro(
            name=f"U{index}",
            kind="M1",
            width=1.0,
            height=1.0,
            x=float(index * 2),
            y=0.0,
            orientation="N",
            is_fixed=False,
        )
        for index in range(20)
    }
    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 80.0, 20.0),
        macros=macros,
        nets=[Net(name="N0", pins=tuple(macros))],
        rows=(
            PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=400, step_x=0.2, step_y=0.0),
            PlacementRow(name="row1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=400, step_x=0.2, step_y=0.0),
            PlacementRow(name="row2", origin_x=0.0, origin_y=4.0, orientation="N", site_count=400, step_x=0.2, step_y=0.0),
        ),
    )

    placement = _build_sample_placement(
        problem=problem,
        graph=None,
        sample={
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_coverage",
            "source": "random_perturbation",
            "move_probability": 0.5,
            "max_x_fraction": 0.1,
            "row_window": 1,
        },
    )

    moved = sum(1 for name, coords in placement.items() if coords != (problem.macros[name].x, problem.macros[name].y))
    assert moved >= 8


def test_sample_macro_perturbation_limits_row_jump_to_local_window():
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net
    from chip.problem import PlacementRow

    class FakeRng:
        def uniform(self, _a, _b):
            return 0.0

        def choice(self, seq):
            return seq[-1]

    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 20.0, 20.0),
        macros={"U1": Macro(name="U1", kind="M1", width=1.0, height=1.0, x=4.0, y=4.0, orientation="N", is_fixed=False)},
        nets=[Net(name="N0", pins=("U1",))],
        rows=(
            PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=100, step_x=0.2, step_y=0.0),
            PlacementRow(name="row1", origin_x=0.0, origin_y=2.0, orientation="FS", site_count=100, step_x=0.2, step_y=0.0),
            PlacementRow(name="row2", origin_x=0.0, origin_y=4.0, orientation="N", site_count=100, step_x=0.2, step_y=0.0),
            PlacementRow(name="row3", origin_x=0.0, origin_y=6.0, orientation="FS", site_count=100, step_x=0.2, step_y=0.0),
            PlacementRow(name="row4", origin_x=0.0, origin_y=8.0, orientation="N", site_count=100, step_x=0.2, step_y=0.0),
        ),
    )

    _next_x, next_y = _sample_macro_perturbation(
        problem=problem,
        macro_name="U1",
        current_x=4.0,
        current_y=4.0,
        rng=FakeRng(),
        x_span=0.4,
        row_window=1,
    )

    assert next_y == 6.0


def test_summarize_dataset_row_keeps_metrics_and_gate_state():
    row = summarize_dataset_row(
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation"},
        metrics={
            "total_routed_wirelength": 100.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 5.0,
            "displacement_median_sites": 2.0,
            "displacement_p90_sites": 6.0,
            "displacement_max_sites": 15.0,
        },
        gate_status="infra_pass / quality_pending",
    )

    assert row["sample_id"] == "rand_0001"
    assert row["total_routed_wirelength"] == 100.0
    assert row["phase1_gate_status"] == "infra_pass / quality_pending"
    assert row["displacement_median_sites"] == 2.0
    assert row["displacement_p90_sites"] == 6.0
    assert row["displacement_max_sites"] == 15.0


def test_summarize_displacement_distribution_reports_site_scaled_stats():
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net
    from chip.problem import PlacementRow

    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 10.0, 10.0),
        macros={
            "U1": Macro(name="U1", kind="CELL", width=0.2, height=2.0, x=0.0, y=0.0, orientation="N", is_fixed=False),
            "U2": Macro(name="U2", kind="CELL", width=0.2, height=2.0, x=1.0, y=0.0, orientation="N", is_fixed=False),
            "U3": Macro(name="U3", kind="CELL", width=0.2, height=2.0, x=2.0, y=0.0, orientation="N", is_fixed=True),
        },
        nets=[Net(name="N1", pins=("U1", "U2", "U3"))],
        rows=(PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=50, step_x=0.2, step_y=0.0),),
    )

    stats = summarize_displacement_distribution(
        problem=problem,
        baseline_placement={"U1": (0.0, 0.0), "U2": (1.0, 0.0), "U3": (2.0, 0.0)},
        candidate_placement={"U1": (0.4, 0.0), "U2": (2.0, 0.0), "U3": (9.0, 0.0)},
    )

    assert stats["displacement_median_sites"] == 3.5
    assert stats["displacement_p90_sites"] == 4.7
    assert stats["displacement_max_sites"] == 5.0


def test_load_phase1_gate_map_reads_benchmark_statuses(tmp_path):
    results_path = tmp_path / "phase1_acceptance_results.json"
    results_path.write_text(
        json.dumps(
            [
                {"benchmark": "mgc_pci_bridge32_1", "phase1_gate_status": "infra_pass / quality_pending"},
                {"benchmark": "mgc_des_perf_2", "phase1_gate_status": "infra_pass / quality_pass"},
            ]
        )
    )

    from chip.experiments.phase2_dataset import load_phase1_gate_map

    gate_map = load_phase1_gate_map(results_path)

    assert gate_map == {
        "mgc_pci_bridge32_1": "infra_pass / quality_pending",
        "mgc_des_perf_2": "infra_pass / quality_pass",
    }


def test_collect_sample_result_marks_timeout_without_losing_metadata():
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "rand_0001",
        "source": "random_perturbation",
        "batch": "smoke",
    }

    def timeout_evaluator(_sample, timeout_seconds):
        raise TimeoutError(f"timed out after {timeout_seconds}s")

    row = collect_sample_result(
        sample=sample,
        gate_status="infra_pass / quality_pending",
        evaluate_fn=timeout_evaluator,
        timeout_seconds=30,
    )

    assert row["status"] == "timeout"
    assert row["sample_id"] == "rand_0001"
    assert row["batch"] == "smoke"
    assert row["phase1_gate_status"] == "infra_pass / quality_pending"
    assert "timed out" in row["error"]


def test_collect_sample_result_marks_errors_without_dropping_sample_identity():
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "sa_0005",
        "source": "sa_truncated",
        "batch": "smoke",
    }

    def failing_evaluator(_sample, timeout_seconds):
        raise RuntimeError(f"router failed inside {timeout_seconds}s window")

    row = collect_sample_result(
        sample=sample,
        gate_status="infra_pass / quality_pending",
        evaluate_fn=failing_evaluator,
        timeout_seconds=45,
    )

    assert row["status"] == "error"
    assert row["sample_id"] == "sa_0005"
    assert row["batch"] == "smoke"
    assert "router failed" in row["error"]


def test_collect_ground_truth_writes_success_rows_and_metrics_artifacts(tmp_path):
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "rand_0001",
        "source": "random_perturbation",
        "batch": "smoke",
        "phase1_gate_status": "infra_pass / quality_pass",
    }

    def fake_evaluator(sample_row, timeout_seconds, artifact_paths):
        assert timeout_seconds == 30
        artifact_paths["placement_def"].write_text("VERSION 5.8 ;\nEND DESIGN\n")
        return {
            "total_routed_wirelength": 123.0,
            "wns": -0.2,
            "tns": -1.5,
            "drc_violations": 4,
            "runtime_seconds": 9.0,
        }

    output_path = tmp_path / "phase2_results.json"
    rows = collect_ground_truth(
        samples=[sample],
        output_path=output_path,
        evaluate_fn=fake_evaluator,
        timeout_seconds=30,
    )

    assert rows[0]["status"] == "success"
    assert rows[0]["placement_def"].endswith("phase2_samples/mgc_pci_bridge32_1/rand_0001.def")
    assert rows[0]["metrics_json"].endswith("phase2_samples/mgc_pci_bridge32_1/rand_0001.metrics.json")
    assert output_path.exists()
    metrics_path = tmp_path / "phase2_samples" / "mgc_pci_bridge32_1" / "rand_0001.metrics.json"
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text())["total_routed_wirelength"] == 123.0


def test_collect_ground_truth_keeps_timeout_rows_in_output(tmp_path):
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "sa_0005",
        "source": "sa_truncated",
        "batch": "smoke",
        "phase1_gate_status": "infra_pass / quality_pending",
    }

    def timeout_evaluator(_sample, timeout_seconds, _artifact_paths):
        raise TimeoutError(f"timed out after {timeout_seconds}s")

    output_path = tmp_path / "phase2_results.json"
    rows = collect_ground_truth(
        samples=[sample],
        output_path=output_path,
        evaluate_fn=timeout_evaluator,
        timeout_seconds=45,
    )

    assert rows[0]["status"] == "timeout"
    assert rows[0]["phase1_gate_status"] == "infra_pass / quality_pending"
    assert json.loads(output_path.read_text())[0]["status"] == "timeout"


def test_collect_ground_truth_persists_rows_between_samples_for_resume(tmp_path):
    samples = [
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0001",
            "source": "random_perturbation",
            "batch": "full",
            "phase1_gate_status": "infra_pass / quality_pass",
        },
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0002",
            "source": "random_perturbation",
            "batch": "full",
            "phase1_gate_status": "infra_pass / quality_pass",
        },
    ]
    output_path = tmp_path / "phase2_results.json"

    def fake_evaluator(sample_row, timeout_seconds, artifact_paths):
        if sample_row["sample_id"] == "rand_0002":
            persisted = json.loads(output_path.read_text())
            assert persisted[0]["sample_id"] == "rand_0001"
        artifact_paths["placement_def"].write_text("VERSION 5.8 ;\nEND DESIGN\n")
        return {
            "total_routed_wirelength": 123.0,
            "wns": -0.2,
            "tns": -1.5,
            "drc_violations": 0,
            "runtime_seconds": 9.0,
        }

    rows = collect_ground_truth(
        samples=samples,
        output_path=output_path,
        evaluate_fn=fake_evaluator,
        timeout_seconds=30,
    )

    assert [row["sample_id"] for row in rows] == ["rand_0001", "rand_0002"]
    assert [row["status"] for row in rows] == ["success", "success"]


def test_collect_ground_truth_skips_existing_rows_when_resuming(tmp_path):
    existing_row = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "rand_0001",
        "source": "random_perturbation",
        "batch": "full",
        "status": "success",
        "phase1_gate_status": "infra_pass / quality_pass",
        "total_routed_wirelength": 111.0,
        "wns": -0.1,
        "tns": -1.0,
        "drc_violations": 0,
        "runtime_seconds": 5.0,
        "placement_def": "already_there.def",
        "metrics_json": "already_there.json",
    }
    output_path = tmp_path / "phase2_results.json"
    output_path.write_text(json.dumps([existing_row], indent=2))
    samples = [
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0001",
            "source": "random_perturbation",
            "batch": "full",
            "phase1_gate_status": "infra_pass / quality_pass",
        },
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0002",
            "source": "random_perturbation",
            "batch": "full",
            "phase1_gate_status": "infra_pass / quality_pass",
        },
    ]
    seen = []

    def fake_evaluator(sample_row, timeout_seconds, artifact_paths):
        seen.append(sample_row["sample_id"])
        artifact_paths["placement_def"].write_text("VERSION 5.8 ;\nEND DESIGN\n")
        return {
            "total_routed_wirelength": 123.0,
            "wns": -0.2,
            "tns": -1.5,
            "drc_violations": 0,
            "runtime_seconds": 9.0,
        }

    rows = collect_ground_truth(
        samples=samples,
        output_path=output_path,
        evaluate_fn=fake_evaluator,
        timeout_seconds=30,
    )

    assert seen == ["rand_0002"]
    assert [row["sample_id"] for row in rows] == ["rand_0001", "rand_0002"]


def test_evaluate_sample_scores_the_legalized_def_without_reinjecting_every_cell(monkeypatch, tmp_path):
    from chip.experiments import phase2_dataset as module
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net
    from chip.problem import PlacementRow

    benchmark_root = tmp_path / "benchmarks"
    artifact_dir = benchmark_root / "mgc_pci_bridge32_1"
    artifact_dir.mkdir(parents=True)
    prepared = {
        "artifact_dir": artifact_dir,
        "tech_lef_path": artifact_dir / "tech.lef",
        "site_fix_lef_path": artifact_dir / "site_fix.lef",
        "cells_lef_path": artifact_dir / "cells.lef",
        "def_path": artifact_dir / "floorplan_clean.def",
        "liberty_path": artifact_dir / "contest.lib",
        "sdc_path": artifact_dir / "acceptance.sdc",
    }
    for key, path in prepared.items():
        if key != "artifact_dir":
            path.write_text("")

    seed_def_path = artifact_dir / "seed_placed.def"
    seed_def_path.write_text("VERSION 5.8 ;\nEND DESIGN\n")
    legalized_def_path = artifact_dir / "rand_0001.def"
    legalized_def_path.write_text("VERSION 5.8 ;\nEND DESIGN\n")

    seed_problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 100.0, 100.0),
        macros={"U1": Macro(name="U1", kind="M1", width=1.0, height=1.0, x=1.0, y=2.0, orientation="N", is_fixed=False)},
        nets=[Net(name="N1", pins=("U1",))],
        rows=(PlacementRow(name="row0", origin_x=0.0, origin_y=2.0, orientation="N", site_count=500, step_x=0.2, step_y=0.0),),
    )
    legalized_problem = ChipProblem(
        die_area=seed_problem.die_area,
        macros={"U1": Macro(name="U1", kind="M1", width=1.0, height=1.0, x=10.2, y=2.0, orientation="N", is_fixed=False)},
        nets=seed_problem.nets,
        rows=seed_problem.rows,
    )

    class FakeGraph:
        node_ids = ("U1",)

    monkeypatch.setattr(module, "prepare_benchmark_artifacts", lambda **_: prepared)
    monkeypatch.setattr(module, "_build_sample_placement", lambda **_: {"U1": (10.2, 2.0)})

    def fake_load_chip_problem(_lef_path, def_path):
        if str(def_path).endswith("rand_0001.def"):
            return FakeGraph(), legalized_problem
        assert str(def_path).endswith("seed_placed.def")
        return FakeGraph(), seed_problem

    monkeypatch.setattr(module, "load_chip_problem", fake_load_chip_problem)
    legalize_calls = []

    def fake_legalize_placement(**kwargs):
        legalize_calls.append(kwargs)
        return legalized_def_path

    monkeypatch.setattr(module, "legalize_placement", fake_legalize_placement)
    captured_requests = []

    def fake_build_evaluation_request(**kwargs):
        captured_requests.append(kwargs)
        return kwargs

    monkeypatch.setattr(module, "build_evaluation_request", fake_build_evaluation_request)

    bridge = object()

    def fake_evaluate_openroad_request(request, **kwargs):
        assert kwargs == {"bridge": bridge, "timeout_seconds": 30}
        return {
            "total_routed_wirelength": 42.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 3.0,
        }

    monkeypatch.setattr(module, "OpenROADBridge", lambda **_: bridge)
    monkeypatch.setattr(module, "evaluate_openroad_request", fake_evaluate_openroad_request)

    artifact_paths = module._sample_artifact_paths(
        samples_root=tmp_path / "phase2_samples",
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001"},
    )
    metrics = module._evaluate_sample(
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation"},
        benchmark_root=benchmark_root,
        artifact_paths=artifact_paths,
        timeout_seconds=30,
    )

    assert metrics["total_routed_wirelength"] == 42.0
    assert legalize_calls[0]["def_path"] == "seed_placed.def"
    assert captured_requests[0]["def_path"] == "rand_0001.def"


def test_build_sample_placement_random_perturbation_uses_sampled_macro_moves(monkeypatch):
    from chip.experiments import phase2_dataset as module
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net
    from chip.problem import PlacementRow

    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 10.0, 10.0),
        macros={
            "U1": Macro(name="U1", kind="CELL", width=0.2, height=2.0, x=0.0, y=0.0, orientation="N", is_fixed=False),
            "U2": Macro(name="U2", kind="CELL", width=0.2, height=2.0, x=0.2, y=0.0, orientation="N", is_fixed=False),
        },
        nets=[Net(name="N1", pins=("U1", "U2"))],
        rows=(PlacementRow(name="row0", origin_x=0.0, origin_y=0.0, orientation="N", site_count=50, step_x=0.2, step_y=0.0),),
    )
    monkeypatch.setattr(
        module,
        "_sample_macro_perturbation",
        lambda problem, macro_name, current_x, current_y, rng, x_span, row_window=1: (current_x + 0.4, current_y),
    )

    placement = module._build_sample_placement(
        problem=problem,
        graph=None,
        sample={
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "rand_0000",
            "source": "random_perturbation",
            "move_probability": 1.0,
        },
    )

    assert placement == {"U1": (0.4, 0.0), "U2": (0.6000000000000001, 0.0)}


def test_build_sample_placement_falls_back_to_initial_when_sa_sample_is_illegal(monkeypatch):
    from chip.experiments import phase2_dataset as module
    from chip.problem import ChipProblem
    from chip.problem import DieArea
    from chip.problem import Macro
    from chip.problem import Net

    problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 10.0, 10.0),
        macros={"U1": Macro(name="U1", kind="CELL", width=0.2, height=2.0, x=0.0, y=0.0, orientation="N", is_fixed=False)},
        nets=[Net(name="N1", pins=("U1",))],
    )
    graph = type("Graph", (), {"global_features": {"bounds": (0.0, 10.0, 0.0, 10.0)}})()
    initial = {"U1": (0.0, 0.0)}
    monkeypatch.setattr(module, "_build_sa_initial_placement", lambda _graph, _problem: initial)
    monkeypatch.setattr(
        module,
        "_run_sa_baseline",
        lambda _problem, *, initial_placement, iterations: ({"U1": (9.8, 9.8)}, {"is_valid": False, "hpwl": 1.0}),
    )

    placement = module._build_sample_placement(
        problem=problem,
        graph=graph,
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "sa_0001", "source": "sa_truncated", "sa_iterations": 1},
    )

    assert placement == initial


def test_ensure_seed_placed_def_runs_built_in_flow_once_when_missing(monkeypatch, tmp_path):
    from chip.experiments import phase2_dataset as module
    from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS

    artifact_dir = tmp_path / "mgc_pci_bridge32_1"
    artifact_dir.mkdir()
    prepared = {
        "artifact_dir": artifact_dir,
        "tech_lef_path": artifact_dir / "tech.lef",
        "site_fix_lef_path": artifact_dir / "site_fix.lef",
        "cells_lef_path": artifact_dir / "cells.lef",
        "def_path": artifact_dir / "floorplan_clean.def",
        "liberty_path": artifact_dir / "contest.lib",
        "sdc_path": artifact_dir / "acceptance.sdc",
    }
    for key, path in prepared.items():
        if key != "artifact_dir":
            path.write_text("")

    captured_requests = []

    def fake_build_evaluation_request(**kwargs):
        captured_requests.append(kwargs)
        return kwargs

    def fake_evaluate_openroad_request(_request, **kwargs):
        (artifact_dir / "seed_placed.def").write_text("VERSION 5.8 ;\nEND DESIGN\n")
        return {
            "total_routed_wirelength": 0.0,
            "wns": 0.0,
            "tns": 0.0,
            "drc_violations": 0,
            "runtime_seconds": 1.0,
        }

    monkeypatch.setattr(module, "build_evaluation_request", fake_build_evaluation_request)
    monkeypatch.setattr(module, "evaluate_openroad_request", fake_evaluate_openroad_request)

    bridge = object()
    seed_path = module._ensure_seed_placed_def(prepared, ACCEPTANCE_BENCHMARKS[0], bridge)

    assert seed_path == artifact_dir / "seed_placed.def"
    assert captured_requests[0]["placement_mode"] == "built_in"
    assert captured_requests[0]["write_def_path"] == "seed_placed.def"

    captured_requests.clear()
    same_seed_path = module._ensure_seed_placed_def(prepared, ACCEPTANCE_BENCHMARKS[0], bridge)
    assert same_seed_path == seed_path
    assert captured_requests == []
