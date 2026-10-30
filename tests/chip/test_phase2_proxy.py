import json

import torch

from chip.experiments.phase2_proxy import build_leave_one_benchmark_out_splits
from chip.experiments.phase2_proxy import build_proxy_run_plan
from chip.experiments.phase2_proxy import build_same_design_splits
from chip.experiments.phase2_proxy import build_graph_level_feature_matrix
from chip.experiments.phase2_proxy import add_seed_relative_node_features
from chip.experiments.phase2_proxy import order_same_design_splits
from chip.experiments.phase2_proxy import fit_graph_stats_linear_model
from chip.experiments.phase2_proxy import predict_graph_stats_linear_model
from chip.experiments.phase2_proxy import select_proxy_rows
from chip.experiments.phase2_proxy import select_active_target_names
from chip.experiments.phase2_proxy import FeatureSchema
from chip.experiments.phase2_proxy import build_validation_report
from chip.experiments.phase2_proxy import encode_problem_graph
from chip.experiments.phase2_proxy import fit_feature_schema
from chip.experiments.phase2_proxy import train_graph_stats_model
from chip.experiments.phase2_proxy import run_phase2_proxy
from chip.experiments.phase2_proxy import train_proxy_model
from chip.problem import ChipProblem
from chip.problem import DieArea
from chip.problem import Macro
from engine.graph import ProblemGraph


def test_fit_feature_schema_keeps_numeric_and_categorical_node_features():
    graphs = [
        ProblemGraph(
            nodes={
                "a": {"width": 1.0, "height": 2.0, "is_fixed": False, "kind": "CELL", "orientation": "N"},
                "b": {"width": 2.0, "height": 2.0, "is_fixed": True, "kind": "PIN", "orientation": "FS"},
            },
            edges=[("a", "b", {"weight": 1.0})],
        )
    ]

    schema = fit_feature_schema(graphs)

    assert schema.numeric_keys == ("height", "is_fixed", "width")
    assert schema.categorical_vocab["kind"] == ("CELL", "PIN")
    assert schema.categorical_vocab["orientation"] == ("FS", "N")


def test_encode_problem_graph_returns_graph_tensors():
    graph = ProblemGraph(
        nodes={
            "a": {"width": 1.0, "height": 2.0, "is_fixed": False, "kind": "CELL", "orientation": "N"},
            "b": {"width": 2.0, "height": 2.0, "is_fixed": True, "kind": "PIN", "orientation": "FS"},
        },
        edges=[("a", "b", {"weight": 1.0})],
    )
    schema = FeatureSchema(
        numeric_keys=("height", "is_fixed", "width"),
        categorical_vocab={"kind": ("CELL", "PIN"), "orientation": ("FS", "N")},
    )

    encoded = encode_problem_graph(graph, schema)

    assert tuple(encoded["node_features"].shape) == (2, 7)
    assert encoded["edge_index"].tolist() == [[0], [1]]


def test_train_proxy_model_overfits_tiny_dataset():
    encoded_graphs = [
        {
            "node_features": torch.tensor([[0.0], [0.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        },
        {
            "node_features": torch.tensor([[1.0], [1.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        },
    ]
    targets = torch.tensor([[0.0], [2.0]])

    model, history = train_proxy_model(
        train_graphs=encoded_graphs,
        train_targets=targets,
        validation_graphs=encoded_graphs,
        validation_targets=targets,
        hidden_dim=16,
        epochs=250,
        learning_rate=0.05,
    )

    predictions = model  # keep reference alive for type check
    assert history["final_train_loss"] < 0.02
    assert history["validation_spearman"] > 0.99
    assert predictions is not None


def test_train_graph_stats_model_overfits_geometry_signal():
    train_rows = [
        {
            "moved_fraction": 0.1,
            "displacement_median_sites": 0.0,
            "displacement_p90_sites": 10.0,
            "displacement_max_sites": 40.0,
        },
        {
            "moved_fraction": 0.2,
            "displacement_median_sites": 1.0,
            "displacement_p90_sites": 20.0,
            "displacement_max_sites": 80.0,
        },
        {
            "moved_fraction": 0.3,
            "displacement_median_sites": 2.0,
            "displacement_p90_sites": 30.0,
            "displacement_max_sites": 120.0,
        },
    ]
    validation_rows = [
        {
            "moved_fraction": 0.15,
            "displacement_median_sites": 0.5,
            "displacement_p90_sites": 15.0,
            "displacement_max_sites": 60.0,
        },
        {
            "moved_fraction": 0.25,
            "displacement_median_sites": 1.5,
            "displacement_p90_sites": 25.0,
            "displacement_max_sites": 100.0,
        },
    ]
    train_targets = torch.tensor([[10.0], [20.0], [30.0]])
    validation_targets = torch.tensor([[15.0], [25.0]])

    _, history, train_predictions, validation_predictions = train_graph_stats_model(
        train_rows=train_rows,
        train_targets=train_targets,
        validation_rows=validation_rows,
        validation_targets=validation_targets,
    )

    assert history["validation_spearman"] > 0.99
    assert train_predictions.shape == train_targets.shape
    assert validation_predictions.shape == validation_targets.shape


def test_build_validation_report_includes_speedup_and_spearman():
    report = build_validation_report(
        target_names=("total_routed_wirelength", "wns"),
        y_true=torch.tensor([[100.0, -0.3], [120.0, -0.5], [80.0, -0.1]]),
        y_pred=torch.tensor([[101.0, -0.31], [119.0, -0.48], [81.0, -0.12]]),
        mean_openroad_runtime_seconds=120.0,
        mean_proxy_runtime_seconds=0.01,
    )

    assert report["mean_spearman"] > 0.9
    assert report["speedup_vs_openroad"] > 1000
    assert set(report["per_target_spearman"]) == {"total_routed_wirelength", "wns"}


def test_build_graph_level_feature_matrix_expands_base_features():
    rows = [
        {
            "moved_fraction": 0.25,
            "displacement_median_sites": 1.0,
            "displacement_p90_sites": 4.0,
            "displacement_max_sites": 9.0,
        }
    ]

    features = build_graph_level_feature_matrix(rows)

    assert tuple(features.shape) == (1, 12)
    assert features[0, 0].item() == 0.25
    assert features[0, 4].item() == 0.25**2


def test_add_seed_relative_node_features_injects_normalized_displacement_channels():
    graph = ProblemGraph(
        nodes={
            "a": {"x": 20.0, "y": 10.0},
            "b": {"x": 40.0, "y": 25.0},
        },
        edges=[],
    )
    sample_problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 100.0, 50.0),
        macros={
            "a": Macro(name="a", kind="CELL", width=1.0, height=1.0, x=20.0, y=10.0),
            "b": Macro(name="b", kind="CELL", width=1.0, height=1.0, x=40.0, y=25.0),
        },
        nets=[],
    )
    seed_problem = ChipProblem(
        die_area=DieArea(0.0, 0.0, 100.0, 50.0),
        macros={
            "a": Macro(name="a", kind="CELL", width=1.0, height=1.0, x=10.0, y=5.0),
            "b": Macro(name="b", kind="CELL", width=1.0, height=1.0, x=40.0, y=25.0),
        },
        nets=[],
    )

    augmented = add_seed_relative_node_features(graph, sample_problem, seed_problem)

    assert augmented.nodes["a"]["dx"] == 0.1
    assert augmented.nodes["a"]["dy"] == 0.1
    assert augmented.nodes["a"]["dx_abs"] == 0.1
    assert augmented.nodes["a"]["dy_abs"] == 0.1
    assert augmented.nodes["a"]["moved_flag"] == 1.0
    assert augmented.nodes["a"]["dr"] > 0.14
    assert augmented.nodes["b"]["dx"] == 0.0
    assert augmented.nodes["b"]["dy"] == 0.0
    assert augmented.nodes["b"]["dr"] == 0.0
    assert augmented.nodes["b"]["moved_flag"] == 0.0


def test_build_leave_one_benchmark_out_splits_holds_out_each_design_once():
    rows = [
        {"benchmark": "mgc_pci_bridge32_1"},
        {"benchmark": "mgc_pci_bridge32_1"},
        {"benchmark": "mgc_pci_bridge32_2"},
        {"benchmark": "mgc_des_perf_2"},
    ]

    splits = build_leave_one_benchmark_out_splits(rows)

    assert [split["holdout_benchmark"] for split in splits] == [
        "mgc_des_perf_2",
        "mgc_pci_bridge32_1",
        "mgc_pci_bridge32_2",
    ]
    assert splits[0]["validation_indices"] == [3]
    assert splits[1]["validation_indices"] == [0, 1]
    assert splits[2]["validation_indices"] == [2]
    assert splits[0]["train_indices"] == [0, 1, 2]


def test_select_proxy_rows_limits_each_benchmark_and_keeps_sa_when_available():
    rows = [
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0000", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0002", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "sa_0001", "source": "sa_truncated", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "sa_0005", "source": "sa_truncated", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "rand_0000", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "rand_0001", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "rand_0002", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "rand_0003", "source": "random_perturbation", "status": "success"},
    ]

    selected = select_proxy_rows(
        rows,
        max_rows_per_benchmark=3,
        allowed_sources=("random_perturbation", "sa_truncated"),
    )

    assert [row["sample_id"] for row in selected if row["benchmark"] == "mgc_pci_bridge32_1"] == [
        "rand_0000",
        "sa_0001",
        "sa_0005",
    ]
    assert len([row for row in selected if row["benchmark"] == "mgc_pci_bridge32_2"]) == 3


def test_select_proxy_rows_defaults_to_random_perturbations_only():
    rows = [
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0000", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "sa_0001", "source": "sa_truncated", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "rand_0000", "source": "random_perturbation", "status": "success"},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "sa_0001", "source": "sa_truncated", "status": "success"},
    ]

    selected = select_proxy_rows(rows, max_rows_per_benchmark=None)

    assert [row["sample_id"] for row in selected] == ["rand_0000", "rand_0001", "rand_0000"]


def test_select_active_target_names_drops_targets_constant_within_each_benchmark():
    rows = [
        {
            "benchmark": "a",
            "sample_id": "a_1",
            "status": "success",
            "hpwl": 10.0,
            "total_routed_wirelength": 100.0,
            "wns": -1.0,
            "tns": -10.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "a",
            "sample_id": "a_2",
            "status": "success",
            "hpwl": 12.0,
            "total_routed_wirelength": 120.0,
            "wns": -1.0,
            "tns": -10.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "b",
            "sample_id": "b_1",
            "status": "success",
            "hpwl": 20.0,
            "total_routed_wirelength": 200.0,
            "wns": -2.0,
            "tns": -20.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "b",
            "sample_id": "b_2",
            "status": "success",
            "hpwl": 21.0,
            "total_routed_wirelength": 210.0,
            "wns": -2.0,
            "tns": -20.0,
            "drc_violations": 0.0,
        },
    ]

    active = select_active_target_names(rows, ("hpwl", "total_routed_wirelength", "wns", "tns", "drc_violations"))

    assert active == ("hpwl", "total_routed_wirelength")


def test_build_same_design_splits_holds_out_validation_rows_inside_each_benchmark():
    rows = [
        {"benchmark": "a", "sample_id": f"a_{index:02d}", "status": "success", "hpwl": float(index)}
        for index in range(10)
    ] + [
        {"benchmark": "b", "sample_id": f"b_{index:02d}", "status": "success", "hpwl": float(index)}
        for index in range(10)
    ]

    splits = build_same_design_splits(rows, validation_fraction=0.2, ranking_target="hpwl")

    assert [split["benchmark"] for split in splits] == ["a", "b"]
    assert all(len(split["validation_indices"]) == 2 for split in splits)
    assert all(len(split["train_indices"]) == 8 for split in splits)
    assert all(set(split["train_indices"]).isdisjoint(split["validation_indices"]) for split in splits)
    assert set(splits[0]["train_indices"] + splits[0]["validation_indices"]) == set(range(10))


def test_order_same_design_splits_prioritizes_largest_hpwl_spread():
    rows = [
        {"benchmark": "mgc_des_perf_2", "sample_id": f"d_{index}", "status": "success", "hpwl": value}
        for index, value in enumerate((1000.0, 1002.0, 1004.0, 1006.0))
    ] + [
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": f"p1_{index}", "status": "success", "hpwl": value}
        for index, value in enumerate((100.0, 110.0, 120.0, 130.0))
    ] + [
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": f"p2_{index}", "status": "success", "hpwl": value}
        for index, value in enumerate((100.0, 107.0, 114.0, 121.0))
    ]
    splits = build_same_design_splits(rows, validation_fraction=0.25, ranking_target="hpwl")

    ordered = order_same_design_splits(splits, rows=rows, ranking_target="hpwl")

    assert [split["benchmark"] for split in ordered] == [
        "mgc_pci_bridge32_1",
        "mgc_pci_bridge32_2",
        "mgc_des_perf_2",
    ]


def test_build_proxy_run_plan_can_target_single_same_design_benchmark():
    rows = [
        {"benchmark": "mgc_des_perf_2", "sample_id": "d0", "status": "success", "hpwl": 1000.0},
        {"benchmark": "mgc_des_perf_2", "sample_id": "d1", "status": "success", "hpwl": 1010.0},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "p10", "status": "success", "hpwl": 100.0},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "p11", "status": "success", "hpwl": 130.0},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "p20", "status": "success", "hpwl": 100.0},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "p21", "status": "success", "hpwl": 120.0},
    ]

    plan = build_proxy_run_plan(rows, mode="same-design", ranking_target="hpwl", benchmark="mgc_pci_bridge32_2")

    assert plan["mode"] == "same-design"
    assert [split["benchmark"] for split in plan["same_design_splits"]] == ["mgc_pci_bridge32_2"]
    assert plan["cross_design_splits"] == []
    assert plan["run_full_fit"] is False


def test_build_proxy_run_plan_can_target_single_cross_design_holdout():
    rows = [
        {"benchmark": "mgc_des_perf_2", "sample_id": "d0", "status": "success", "hpwl": 1000.0},
        {"benchmark": "mgc_des_perf_2", "sample_id": "d1", "status": "success", "hpwl": 1010.0},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "p10", "status": "success", "hpwl": 100.0},
        {"benchmark": "mgc_pci_bridge32_1", "sample_id": "p11", "status": "success", "hpwl": 130.0},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "p20", "status": "success", "hpwl": 100.0},
        {"benchmark": "mgc_pci_bridge32_2", "sample_id": "p21", "status": "success", "hpwl": 120.0},
    ]

    plan = build_proxy_run_plan(rows, mode="cross-design", ranking_target="hpwl", holdout_benchmark="mgc_des_perf_2")

    assert plan["same_design_splits"] == []
    assert [split["holdout_benchmark"] for split in plan["cross_design_splits"]] == ["mgc_des_perf_2"]
    assert plan["run_full_fit"] is False


def test_run_phase2_proxy_reports_cross_design_folds(monkeypatch, tmp_path):
    graphs = [
        ProblemGraph(nodes={"a1": {"width": 1.0}}, edges=[]),
        ProblemGraph(nodes={"a2": {"width": 1.5}}, edges=[]),
        ProblemGraph(nodes={"b1": {"width": 2.0}}, edges=[]),
        ProblemGraph(nodes={"b2": {"width": 2.5}}, edges=[]),
    ]
    rows = [
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "a1",
            "status": "success",
            "runtime_seconds": 10.0,
            "hpwl": 100.0,
            "total_routed_wirelength": -0.1,
            "wns": -1.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "a2",
            "status": "success",
            "runtime_seconds": 11.0,
            "hpwl": 120.0,
            "total_routed_wirelength": -0.2,
            "wns": -1.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_pci_bridge32_2",
            "sample_id": "b1",
            "status": "success",
            "runtime_seconds": 12.0,
            "hpwl": 140.0,
            "total_routed_wirelength": -0.3,
            "wns": -2.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_pci_bridge32_2",
            "sample_id": "b2",
            "status": "success",
            "runtime_seconds": 13.0,
            "hpwl": 160.0,
            "total_routed_wirelength": -0.4,
            "wns": -2.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
    ]

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_rows",
        lambda dataset_results_path, max_rows_per_benchmark=None, allowed_sources=None: rows,
    )
    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_graphs",
        lambda selected_rows, benchmark_root, target_names: (
            [graphs[rows.index(row)] for row in selected_rows],
            torch.tensor([[float(row[name]) for name in target_names] for row in selected_rows], dtype=torch.float32),
        ),
    )

    class StaticModel(torch.nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        def forward(self, batch):
            return torch.zeros((batch.num_graphs, self.output_dim), dtype=torch.float32)

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.train_proxy_model",
        lambda **kwargs: (
            StaticModel(kwargs["train_targets"].shape[1]),
            {"final_train_loss": 0.01, "validation_spearman": 0.25},
        ),
    )
    dataset_path = tmp_path / "phase2_dataset_results.json"
    dataset_path.write_text(json.dumps(rows))

    result = run_phase2_proxy(
        dataset_results_path=dataset_path,
        benchmark_root=tmp_path / "benchmarks",
        output_dir=tmp_path / "phase2_proxy",
        epochs=5,
        mode="all",
    )

    assert len(result["report"]["cross_design_folds"]) == 2
    assert len(result["report"]["same_design_folds"]) == 2
    assert {fold["holdout_benchmark"] for fold in result["report"]["cross_design_folds"]} == {
        "mgc_pci_bridge32_1",
        "mgc_pci_bridge32_2",
    }
    assert result["report"]["cross_design_mean_spearman"] == sum(
        fold["mean_spearman"] for fold in result["report"]["cross_design_folds"]
    ) / 2
    assert result["report"]["full_training_row_count"] == 4
    assert result["report"]["active_target_names"] == ["hpwl", "total_routed_wirelength"]
    assert result["report"]["phase2_gate"]["same_design_status"] in {"pass", "fail"}
    assert result["report"]["phase2_gate"]["cross_design_status"] in {"pass", "fail"}


def test_run_phase2_proxy_same_design_mode_writes_only_requested_fold(monkeypatch, tmp_path):
    rows = [
        {
            "benchmark": "mgc_des_perf_2",
            "sample_id": "d0",
            "status": "success",
            "runtime_seconds": 15.0,
            "hpwl": 1000.0,
            "total_routed_wirelength": 1100.0,
            "wns": -1.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_des_perf_2",
            "sample_id": "d1",
            "status": "success",
            "runtime_seconds": 16.0,
            "hpwl": 1010.0,
            "total_routed_wirelength": 1110.0,
            "wns": -1.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "p10",
            "status": "success",
            "runtime_seconds": 10.0,
            "hpwl": 100.0,
            "total_routed_wirelength": 120.0,
            "wns": -2.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
        {
            "benchmark": "mgc_pci_bridge32_1",
            "sample_id": "p11",
            "status": "success",
            "runtime_seconds": 11.0,
            "hpwl": 130.0,
            "total_routed_wirelength": 150.0,
            "wns": -2.0,
            "tns": 0.0,
            "drc_violations": 0.0,
        },
    ]
    graphs_by_sample = {
        "d0": ProblemGraph(nodes={"d0": {"width": 1.0}}, edges=[]),
        "d1": ProblemGraph(nodes={"d1": {"width": 2.0}}, edges=[]),
        "p10": ProblemGraph(nodes={"p10": {"width": 3.0}}, edges=[]),
        "p11": ProblemGraph(nodes={"p11": {"width": 4.0}}, edges=[]),
    }

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_rows",
        lambda dataset_results_path, max_rows_per_benchmark=None, allowed_sources=None: rows,
    )
    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_graphs",
        lambda selected_rows, benchmark_root, target_names: (
            [graphs_by_sample[row["sample_id"]] for row in selected_rows],
            torch.tensor([[float(row[name]) for name in target_names] for row in selected_rows], dtype=torch.float32),
        ),
    )

    class StaticModel(torch.nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim

        def forward(self, batch):
            return torch.zeros((batch.num_graphs, self.output_dim), dtype=torch.float32)

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.train_proxy_model",
        lambda **kwargs: (
            StaticModel(kwargs["train_targets"].shape[1]),
            {"final_train_loss": 0.01, "validation_spearman": 0.25},
        ),
    )

    result = run_phase2_proxy(
        dataset_results_path=tmp_path / "phase2_dataset_results.json",
        benchmark_root=tmp_path / "benchmarks",
        output_dir=tmp_path / "phase2_proxy_same_design",
        epochs=5,
        mode="same-design",
        benchmark="mgc_pci_bridge32_1",
    )

    assert [fold["benchmark"] for fold in result["report"]["same_design_folds"]] == ["mgc_pci_bridge32_1"]
    assert result["report"]["cross_design_folds"] == []
    assert result["report"]["full_training_row_count"] == 2
    assert result["model_path"] is None


def test_run_phase2_proxy_graph_stats_same_design_uses_row_level_geometry(monkeypatch, tmp_path):
    rows = []
    for index in range(10):
        rows.append(
            {
                "benchmark": "mgc_pci_bridge32_1",
                "sample_id": f"p{index:02d}",
                "source": "random_perturbation",
                "status": "success",
                "runtime_seconds": 5.0 + index,
                "hpwl": 100.0 + (index * 10.0),
                "total_routed_wirelength": 120.0 + (index * 10.0),
                "wns": -1.0,
                "tns": 0.0,
                "drc_violations": 0.0,
                "moved_fraction": 0.1 + (index * 0.01),
                "displacement_median_sites": float(index),
                "displacement_p90_sites": 10.0 + (index * 5.0),
                "displacement_max_sites": 20.0 + (index * 10.0),
            }
        )

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_rows",
        lambda dataset_results_path, max_rows_per_benchmark=None, allowed_sources=None: rows,
    )
    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_graphs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph loader should not run for graph-stats")),
    )

    result = run_phase2_proxy(
        dataset_results_path=tmp_path / "phase2_dataset_results.json",
        benchmark_root=tmp_path / "benchmarks",
        output_dir=tmp_path / "phase2_proxy_graph_stats_same_design",
        mode="same-design",
        model_family="graph-stats",
        benchmark="mgc_pci_bridge32_1",
    )

    fold = result["report"]["same_design_folds"][0]
    assert fold["benchmark"] == "mgc_pci_bridge32_1"
    assert fold["mean_spearman"] > 0.9
    assert fold["train_mean_spearman"] > 0.9


def test_run_phase2_proxy_frozen_linear_base_alpha_zero_uses_fold_train_only_and_skips_graph_loader(
    monkeypatch, tmp_path
):
    rows = []
    for index in range(10):
        rows.append(
            {
                "benchmark": "mgc_pci_bridge32_1",
                "sample_id": f"p{index:02d}",
                "source": "random_perturbation",
                "status": "success",
                "runtime_seconds": 5.0 + index,
                "hpwl": 100.0 + (index * 10.0),
                "total_routed_wirelength": 120.0 + (index * 10.0),
                "wns": -1.0,
                "tns": 0.0,
                "drc_violations": 0.0,
                "moved_fraction": 0.1 + (index * 0.01),
                "displacement_median_sites": float(index),
                "displacement_p90_sites": 10.0 + (index * 5.0),
                "displacement_max_sites": 20.0 + (index * 10.0),
            }
        )

    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_rows",
        lambda dataset_results_path, max_rows_per_benchmark=None, allowed_sources=None: rows,
    )
    monkeypatch.setattr(
        "chip.experiments.phase2_proxy.load_phase2_proxy_graphs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph loader should not run for alpha=0 base-only")),
    )

    result = run_phase2_proxy(
        dataset_results_path=tmp_path / "phase2_dataset_results.json",
        benchmark_root=tmp_path / "benchmarks",
        output_dir=tmp_path / "phase2_proxy_frozen_linear_base_same_design",
        mode="same-design",
        model_family="frozen-linear-base",
        benchmark="mgc_pci_bridge32_1",
        residual_alpha=0.0,
    )

    fold = result["report"]["same_design_folds"][0]
    assert fold["benchmark"] == "mgc_pci_bridge32_1"
    assert fold["mean_spearman"] > 0.9
    assert fold["train_mean_spearman"] > 0.9
    assert fold["residual_alpha"] == 0.0
    assert fold["base_mean_spearman"] > 0.9


def test_fold_aware_graph_stats_base_differs_from_full_fit_predictions():
    rows = []
    for index in range(10):
        rows.append(
            {
                "benchmark": "mgc_pci_bridge32_1",
                "sample_id": f"p{index:02d}",
                "source": "random_perturbation",
                "status": "success",
                "runtime_seconds": 5.0 + index,
                "hpwl": 100.0 + (index * 10.0),
                "total_routed_wirelength": 120.0 + (index * 10.0),
                "wns": -1.0,
                "tns": 0.0,
                "drc_violations": 0.0,
                "moved_fraction": 0.1 + (index * 0.01),
                "displacement_median_sites": float(index),
                "displacement_p90_sites": 10.0 + (index * 5.0),
                "displacement_max_sites": 20.0 + (index * 10.0),
            }
        )

    split = build_same_design_splits(rows, validation_fraction=0.2, ranking_target="hpwl")[0]
    train_rows = [rows[index] for index in split["train_indices"]]
    validation_rows = [rows[index] for index in split["validation_indices"]]
    train_targets = torch.tensor([[float(row["hpwl"]), float(row["total_routed_wirelength"])] for row in train_rows])
    all_targets = torch.tensor([[float(row["hpwl"]), float(row["total_routed_wirelength"])] for row in rows])

    fold_model = fit_graph_stats_linear_model(train_rows, train_targets)
    full_model = fit_graph_stats_linear_model(rows, all_targets)

    fold_predictions = predict_graph_stats_linear_model(fold_model, validation_rows)
    full_predictions = predict_graph_stats_linear_model(full_model, validation_rows)

    assert not torch.allclose(fold_predictions, full_predictions)


def test_fold_aware_graph_stats_base_differs_across_same_design_folds():
    rows = []
    for index in range(12):
        rows.append(
            {
                "benchmark": "mgc_pci_bridge32_1",
                "sample_id": f"p{index:02d}",
                "source": "random_perturbation",
                "status": "success",
                "runtime_seconds": 5.0 + index,
                "hpwl": 100.0 + (index * 10.0),
                "total_routed_wirelength": 120.0 + (index * 12.0),
                "wns": -1.0,
                "tns": 0.0,
                "drc_violations": 0.0,
                "moved_fraction": 0.1 + (index * 0.015),
                "displacement_median_sites": float(index),
                "displacement_p90_sites": 10.0 + (index * 4.0),
                "displacement_max_sites": 20.0 + (index * 9.0),
            }
        )

    split_a = {
        "benchmark": "mgc_pci_bridge32_1",
        "train_indices": list(range(0, 8)),
        "validation_indices": list(range(8, 12)),
    }
    split_b = {
        "benchmark": "mgc_pci_bridge32_1",
        "train_indices": list(range(4, 12)),
        "validation_indices": list(range(0, 4)),
    }

    train_rows_a = [rows[index] for index in split_a["train_indices"]]
    train_rows_b = [rows[index] for index in split_b["train_indices"]]
    validation_rows_shared = rows[4:8]
    train_targets_a = torch.tensor([[float(row["hpwl"]), float(row["total_routed_wirelength"])] for row in train_rows_a])
    train_targets_b = torch.tensor([[float(row["hpwl"]), float(row["total_routed_wirelength"])] for row in train_rows_b])

    model_a = fit_graph_stats_linear_model(train_rows_a, train_targets_a)
    model_b = fit_graph_stats_linear_model(train_rows_b, train_targets_b)

    predictions_a = predict_graph_stats_linear_model(model_a, validation_rows_shared)
    predictions_b = predict_graph_stats_linear_model(model_b, validation_rows_shared)

    assert not torch.allclose(predictions_a, predictions_b)
