from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from chip.ispd_benchmarks import resolve_artifact_paths
from chip.netlist_parser import load_chip_problem
from chip.problem import ChipProblem
from engine.gnn_proxy import GNNProxy
from engine.gnn_proxy import batch_graph_tensors
from engine.graph import ProblemGraph


DEFAULT_TARGET_NAMES = ("hpwl", "total_routed_wirelength", "wns", "tns", "drc_violations")
DEFAULT_MAX_ROWS_PER_BENCHMARK = None
DEFAULT_BATCH_SIZE = 2
DEFAULT_SAME_DESIGN_VALIDATION_FRACTION = 0.2
DEFAULT_SAME_DESIGN_SPEARMAN_THRESHOLD = 0.8
DEFAULT_CROSS_DESIGN_SANITY_THRESHOLD = 0.0
DEFAULT_CROSS_DESIGN_STRETCH_THRESHOLD = 0.5
DEFAULT_PROXY_MODE = "all"
DEFAULT_ALLOWED_SOURCES = ("random_perturbation",)
DEFAULT_PROXY_MODEL_FAMILY = "gnn"
DEFAULT_PROXY_POOLING_MODE = "mean"
DEFAULT_NODE_PATH_SCALE = 1.0
DEFAULT_FROZEN_BASE_ALPHA = 0.0
GRAPH_LEVEL_FEATURE_NAMES = (
    "moved_fraction",
    "displacement_median_sites",
    "displacement_p90_sites",
    "displacement_max_sites",
)
SEED_RELATIVE_NODE_FEATURE_NAMES = (
    "dx",
    "dy",
    "dx_abs",
    "dy_abs",
    "dr",
    "moved_flag",
)


@dataclass(frozen=True)
class FeatureSchema:
    numeric_keys: tuple[str, ...]
    categorical_vocab: dict[str, tuple[str, ...]]


def fit_feature_schema(graphs: list[ProblemGraph]) -> FeatureSchema:
    numeric_keys = set()
    categorical_values: dict[str, set[str]] = {}
    for graph in graphs:
        for node_features in graph.nodes.values():
            for key, value in node_features.items():
                if isinstance(value, bool) or isinstance(value, (int, float)):
                    numeric_keys.add(key)
                elif isinstance(value, str):
                    categorical_values.setdefault(key, set()).add(value)
    return FeatureSchema(
        numeric_keys=tuple(sorted(numeric_keys)),
        categorical_vocab={key: tuple(sorted(values)) for key, values in categorical_values.items()},
    )


def encode_problem_graph(graph: ProblemGraph, schema: FeatureSchema) -> dict[str, torch.Tensor]:
    node_ids = list(graph.node_ids)
    node_features = []
    for node_id in node_ids:
        features = graph.nodes[node_id]
        encoded = [float(features.get(key, 0.0)) for key in schema.numeric_keys]
        for key, vocab in schema.categorical_vocab.items():
            value = str(features.get(key, ""))
            encoded.extend(1.0 if category == value else 0.0 for category in vocab)
        node_features.append(encoded)

    index_by_node = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges = [(index_by_node[source], index_by_node[target]) for source, target, _ in graph.edges]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": edge_index,
    }


def add_seed_relative_node_features(
    graph: ProblemGraph,
    problem: ChipProblem,
    seed_problem: ChipProblem,
    *,
    epsilon: float = 1e-9,
) -> ProblemGraph:
    die_width = max(problem.die_area.width, epsilon)
    die_height = max(problem.die_area.height, epsilon)
    for node_id, node_features in graph.nodes.items():
        sample_macro = problem.macros.get(node_id)
        seed_macro = seed_problem.macros.get(node_id)
        if sample_macro is None or seed_macro is None:
            dx = 0.0
            dy = 0.0
        else:
            dx = (sample_macro.x - seed_macro.x) / die_width
            dy = (sample_macro.y - seed_macro.y) / die_height
        dx_abs = abs(dx)
        dy_abs = abs(dy)
        dr = ((dx * dx) + (dy * dy)) ** 0.5
        node_features["dx"] = dx
        node_features["dy"] = dy
        node_features["dx_abs"] = dx_abs
        node_features["dy_abs"] = dy_abs
        node_features["dr"] = dr
        node_features["moved_flag"] = 1.0 if (dx_abs > epsilon or dy_abs > epsilon) else 0.0
    return graph


def extract_graph_level_features(row: dict[str, object]) -> torch.Tensor:
    base = torch.tensor([float(row.get(name, 0.0)) for name in GRAPH_LEVEL_FEATURE_NAMES], dtype=torch.float32)
    return torch.cat([base, base.square(), torch.log1p(base.clamp_min(0.0))], dim=0)


def build_graph_level_feature_matrix(rows: list[dict[str, object]]) -> torch.Tensor:
    if not rows:
        return torch.empty((0, len(GRAPH_LEVEL_FEATURE_NAMES) * 3), dtype=torch.float32)
    return torch.stack([extract_graph_level_features(row) for row in rows], dim=0)


def _normalize_feature_matrix(
    train_features: torch.Tensor,
    other_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        (train_features - feature_mean) / feature_std,
        (other_features - feature_mean) / feature_std,
        feature_mean,
        feature_std,
    )


def _attach_graph_level_features(
    train_graphs: list[dict[str, torch.Tensor]],
    train_rows: list[dict[str, object]],
    validation_graphs: list[dict[str, torch.Tensor]],
    validation_rows: list[dict[str, object]],
) -> tuple[torch.Tensor, torch.Tensor]:
    train_matrix, validation_matrix, feature_mean, feature_std = _normalize_feature_matrix(
        build_graph_level_feature_matrix(train_rows),
        build_graph_level_feature_matrix(validation_rows),
    )
    for graph, features in zip(train_graphs, train_matrix):
        graph["graph_features"] = features
    for graph, features in zip(validation_graphs, validation_matrix):
        graph["graph_features"] = features
    return feature_mean, feature_std


def compute_per_target_spearman(
    target_names: tuple[str, ...],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    per_target = {}
    for index, name in enumerate(target_names):
        true_values = y_true[:, index].cpu().numpy()
        pred_values = y_pred[:, index].cpu().numpy()
        if len(set(true_values.tolist())) < 2 or len(set(pred_values.tolist())) < 2:
            coefficient = 0.0
        else:
            coefficient, _ = spearmanr(true_values, pred_values)
            coefficient = 0.0 if coefficient != coefficient else float(coefficient)
        per_target[name] = float(coefficient)
    return per_target


def train_proxy_model(
    train_graphs: list[dict[str, torch.Tensor]],
    train_targets: torch.Tensor,
    validation_graphs: list[dict[str, torch.Tensor]],
    validation_targets: torch.Tensor,
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pooling_mode: str = DEFAULT_PROXY_POOLING_MODE,
    node_path_scale: float = DEFAULT_NODE_PATH_SCALE,
) -> tuple[GNNProxy, dict[str, float]]:
    model = GNNProxy(
        node_feature_dim=train_graphs[0]["node_features"].shape[1],
        graph_feature_dim=train_graphs[0].get("graph_features", torch.empty((0,))).shape[0],
        hidden_dim=hidden_dim,
        output_dim=train_targets.shape[1],
        num_layers=num_layers,
        pooling_mode=pooling_mode,
        node_path_scale=node_path_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    target_mean = train_targets.mean(dim=0, keepdim=True)
    target_std = train_targets.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    normalized_train_targets = (train_targets - target_mean) / target_std

    for _ in range(epochs):
        last_loss = None
        for batch_indices in _iter_batch_indices(len(train_graphs), batch_size):
            graph_batch = [train_graphs[index] for index in batch_indices]
            target_batch = normalized_train_targets[batch_indices]
            predictions = model(batch_graph_tensors(graph_batch))
            normalized_predictions = (predictions - target_mean) / target_std
            loss = F.mse_loss(normalized_predictions, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss

    with torch.no_grad():
        validation_predictions = _predict_in_batches(model, validation_graphs, batch_size=batch_size)
    report = build_validation_report(
        target_names=tuple(f"target_{index}" for index in range(validation_targets.shape[1])),
        y_true=validation_targets,
        y_pred=validation_predictions,
        mean_openroad_runtime_seconds=1.0,
        mean_proxy_runtime_seconds=0.001,
    )
    report["final_train_loss"] = float(last_loss.item()) if last_loss is not None else 0.0
    report["validation_spearman"] = float(report["mean_spearman"])
    return model, report


def train_graph_stats_model(
    train_rows: list[dict[str, object]],
    train_targets: torch.Tensor,
    validation_rows: list[dict[str, object]],
    validation_targets: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float], torch.Tensor, torch.Tensor]:
    model = fit_graph_stats_linear_model(train_rows, train_targets)
    train_predictions = predict_graph_stats_linear_model(model, train_rows)
    validation_predictions = predict_graph_stats_linear_model(model, validation_rows)
    target_names = tuple(f"target_{index}" for index in range(validation_targets.shape[1]))
    report = build_validation_report(
        target_names=target_names,
        y_true=validation_targets,
        y_pred=validation_predictions,
        mean_openroad_runtime_seconds=1.0,
        mean_proxy_runtime_seconds=0.001,
    )
    report["final_train_loss"] = float(torch.nn.functional.mse_loss(train_predictions, train_targets).item())
    report["validation_spearman"] = float(report["mean_spearman"])
    return (model, report, train_predictions, validation_predictions)


def train_frozen_linear_base_model(
    train_rows: list[dict[str, object]],
    train_targets: torch.Tensor,
    validation_rows: list[dict[str, object]],
    validation_targets: torch.Tensor,
    *,
    benchmark_root: str | Path,
    target_names: tuple[str, ...],
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    pooling_mode: str,
    node_path_scale: float,
    residual_alpha: float,
) -> tuple[dict[str, object], dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = fit_graph_stats_linear_model(train_rows, train_targets)
    base_train_predictions = predict_graph_stats_linear_model(base_model, train_rows)
    base_validation_predictions = predict_graph_stats_linear_model(base_model, validation_rows)
    if abs(residual_alpha) <= 1e-12:
        history = {"final_train_loss": float(torch.nn.functional.mse_loss(base_train_predictions, train_targets).item())}
        return (
            {
                "base_model": base_model,
                "residual_model": None,
                "residual_alpha": residual_alpha,
                "schema": None,
                "graph_feature_mean": None,
                "graph_feature_std": None,
            },
            history,
            base_train_predictions,
            base_validation_predictions,
            base_train_predictions,
            base_validation_predictions,
        )

    train_graphs_raw, _ = load_phase2_proxy_graphs(train_rows, benchmark_root, target_names=target_names)
    validation_graphs_raw, _ = load_phase2_proxy_graphs(validation_rows, benchmark_root, target_names=target_names)
    schema = fit_feature_schema(train_graphs_raw)
    train_graphs = [encode_problem_graph(graph, schema) for graph in train_graphs_raw]
    validation_graphs = [encode_problem_graph(graph, schema) for graph in validation_graphs_raw]
    graph_feature_mean, graph_feature_std = _attach_graph_level_features(
        train_graphs,
        train_rows,
        validation_graphs,
        validation_rows,
    )
    residual_train_targets = train_targets - base_train_predictions
    residual_validation_targets = validation_targets - base_validation_predictions
    residual_model, history = train_proxy_model(
        train_graphs=train_graphs,
        train_targets=residual_train_targets,
        validation_graphs=validation_graphs,
        validation_targets=residual_validation_targets,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        pooling_mode=pooling_mode,
        node_path_scale=node_path_scale,
    )
    with torch.no_grad():
        residual_train_predictions = _predict_in_batches(residual_model, train_graphs, batch_size=batch_size)
        residual_validation_predictions = _predict_in_batches(residual_model, validation_graphs, batch_size=batch_size)
    return (
        {
            "base_model": base_model,
            "residual_model": residual_model,
            "residual_alpha": residual_alpha,
            "schema": schema,
            "graph_feature_mean": graph_feature_mean,
            "graph_feature_std": graph_feature_std,
        },
        history,
        base_train_predictions + (residual_alpha * residual_train_predictions),
        base_validation_predictions + (residual_alpha * residual_validation_predictions),
        base_train_predictions,
        base_validation_predictions,
    )


def fit_graph_stats_linear_model(
    train_rows: list[dict[str, object]],
    train_targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    train_features = build_graph_level_feature_matrix(train_rows)
    normalized_train_features, _, feature_mean, feature_std = _normalize_feature_matrix(train_features, train_features)
    train_design = torch.cat(
        [normalized_train_features, torch.ones((normalized_train_features.shape[0], 1), dtype=torch.float32)],
        dim=1,
    )
    weights = torch.linalg.lstsq(train_design, train_targets).solution
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "weights": weights,
    }


def predict_graph_stats_linear_model(
    model: dict[str, torch.Tensor],
    rows: list[dict[str, object]],
) -> torch.Tensor:
    features = build_graph_level_feature_matrix(rows)
    normalized_features = (features - model["feature_mean"]) / model["feature_std"]
    design = torch.cat([normalized_features, torch.ones((normalized_features.shape[0], 1), dtype=torch.float32)], dim=1)
    return design @ model["weights"]


def build_validation_report(
    target_names: tuple[str, ...],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mean_openroad_runtime_seconds: float,
    mean_proxy_runtime_seconds: float,
) -> dict[str, object]:
    per_target = compute_per_target_spearman(target_names, y_true, y_pred)
    return {
        "per_target_spearman": per_target,
        "mean_spearman": sum(per_target.values()) / len(per_target),
        "mean_openroad_runtime_seconds": mean_openroad_runtime_seconds,
        "mean_proxy_runtime_seconds": mean_proxy_runtime_seconds,
        "speedup_vs_openroad": mean_openroad_runtime_seconds / max(mean_proxy_runtime_seconds, 1e-9),
    }


def build_leave_one_benchmark_out_splits(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    benchmark_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        benchmark = str(row["benchmark"])
        benchmark_to_indices.setdefault(benchmark, []).append(index)

    all_indices = list(range(len(rows)))
    splits: list[dict[str, object]] = []
    for benchmark in sorted(benchmark_to_indices):
        validation_indices = benchmark_to_indices[benchmark]
        train_indices = [index for index in all_indices if index not in validation_indices]
        if train_indices and validation_indices:
            splits.append(
                {
                    "holdout_benchmark": benchmark,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                }
            )
    return splits


def build_same_design_splits(
    rows: list[dict[str, object]],
    validation_fraction: float = DEFAULT_SAME_DESIGN_VALIDATION_FRACTION,
    ranking_target: str | None = None,
) -> list[dict[str, object]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    benchmark_to_indexed_rows: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for index, row in enumerate(rows):
        if row.get("status") != "success":
            continue
        benchmark = str(row["benchmark"])
        benchmark_to_indexed_rows.setdefault(benchmark, []).append((index, row))

    splits: list[dict[str, object]] = []
    for benchmark in sorted(benchmark_to_indexed_rows):
        indexed_rows = benchmark_to_indexed_rows[benchmark]
        if len(indexed_rows) < 2:
            continue
        if ranking_target is not None:
            indexed_rows = sorted(
                indexed_rows,
                key=lambda item: (float(item[1].get(ranking_target, 0.0)), str(item[1].get("sample_id", ""))),
            )
        else:
            indexed_rows = sorted(indexed_rows, key=lambda item: str(item[1].get("sample_id", "")))
        validation_count = min(max(1, round(len(indexed_rows) * validation_fraction)), len(indexed_rows) - 1)
        validation_pairs = _evenly_spaced_take(indexed_rows, validation_count)
        validation_indices = [index for index, _ in validation_pairs]
        train_indices = [index for index, _ in indexed_rows if index not in set(validation_indices)]
        splits.append(
            {
                "benchmark": benchmark,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
            }
        )
    return splits


def order_same_design_splits(
    splits: list[dict[str, object]],
    rows: list[dict[str, object]],
    ranking_target: str,
) -> list[dict[str, object]]:
    spread_by_benchmark: dict[str, float] = {}
    for benchmark in {str(row["benchmark"]) for row in rows if row.get("status") == "success"}:
        values = [float(row[ranking_target]) for row in rows if row.get("status") == "success" and str(row["benchmark"]) == benchmark]
        spread_by_benchmark[benchmark] = (max(values) - min(values)) / max(mean(values), 1e-9) if values else 0.0
    return sorted(
        splits,
        key=lambda split: (-spread_by_benchmark.get(str(split["benchmark"]), 0.0), str(split["benchmark"])),
    )


def select_proxy_rows(
    rows: list[dict[str, object]],
    max_rows_per_benchmark: int | None = DEFAULT_MAX_ROWS_PER_BENCHMARK,
    allowed_sources: tuple[str, ...] = DEFAULT_ALLOWED_SOURCES,
) -> list[dict[str, object]]:
    success_rows = [
        row
        for row in rows
        if row.get("status") == "success" and str(row.get("source", "")) in allowed_sources
    ]
    if max_rows_per_benchmark is None:
        return success_rows

    selected: list[dict[str, object]] = []
    for benchmark in sorted({str(row["benchmark"]) for row in success_rows}):
        benchmark_rows = [row for row in success_rows if str(row["benchmark"]) == benchmark]
        sa_rows = [row for row in benchmark_rows if row.get("source") == "sa_truncated"]
        random_rows = [row for row in benchmark_rows if row.get("source") != "sa_truncated"]
        sa_quota = min(len(sa_rows), max_rows_per_benchmark - (max_rows_per_benchmark // 2))
        random_quota = max_rows_per_benchmark - sa_quota
        if len(random_rows) < random_quota:
            sa_quota = min(len(sa_rows), max_rows_per_benchmark - len(random_rows))
            random_quota = min(len(random_rows), max_rows_per_benchmark - sa_quota)
        selected_ids = {
            id(row)
            for row in _evenly_spaced_take(random_rows, random_quota) + _evenly_spaced_take(sa_rows, sa_quota)
        }
        selected.extend(row for row in benchmark_rows if id(row) in selected_ids)
    return selected


def select_active_target_names(
    rows: list[dict[str, object]],
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
) -> tuple[str, ...]:
    success_rows = [row for row in rows if row.get("status") == "success"]
    active_targets: list[str] = []
    for target_name in target_names:
        varying_benchmark_count = 0
        for benchmark in sorted({str(row["benchmark"]) for row in success_rows}):
            benchmark_values = [row.get(target_name) for row in success_rows if str(row["benchmark"]) == benchmark]
            if len(set(benchmark_values)) > 1:
                varying_benchmark_count += 1
        if varying_benchmark_count > 0:
            active_targets.append(target_name)
    return tuple(active_targets)


def summarize_target_signal(
    rows: list[dict[str, object]],
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
) -> dict[str, object]:
    success_rows = [row for row in rows if row.get("status") == "success"]
    benchmarks = sorted({str(row["benchmark"]) for row in success_rows})
    summary: dict[str, object] = {
        "row_count": len(success_rows),
        "benchmarks": {},
    }
    for benchmark in benchmarks:
        benchmark_rows = [row for row in success_rows if str(row["benchmark"]) == benchmark]
        benchmark_summary: dict[str, object] = {
            "row_count": len(benchmark_rows),
            "source_counts": {},
            "targets": {},
        }
        for row in benchmark_rows:
            source = str(row.get("source", "unknown"))
            benchmark_summary["source_counts"][source] = benchmark_summary["source_counts"].get(source, 0) + 1
        for target_name in target_names:
            values = [float(row[target_name]) for row in benchmark_rows if row.get(target_name) is not None]
            if not values:
                continue
            benchmark_summary["targets"][target_name] = {
                "unique_count": len(set(values)),
                "min": min(values),
                "max": max(values),
                "spread_fraction": (max(values) - min(values)) / max(mean(values), 1e-9),
            }
        if {"hpwl", "total_routed_wirelength"}.issubset(benchmark_summary["targets"]):
            hpwl_values = [float(row["hpwl"]) for row in benchmark_rows]
            routed_values = [float(row["total_routed_wirelength"]) for row in benchmark_rows]
            if len(set(hpwl_values)) > 1 and len(set(routed_values)) > 1:
                coefficient, _ = spearmanr(hpwl_values, routed_values)
                benchmark_summary["hpwl_vs_routed_spearman"] = 0.0 if coefficient != coefficient else float(coefficient)
            else:
                benchmark_summary["hpwl_vs_routed_spearman"] = 0.0
        summary["benchmarks"][benchmark] = benchmark_summary
    return summary


def load_phase2_proxy_rows(
    dataset_results_path: str | Path,
    max_rows_per_benchmark: int | None = DEFAULT_MAX_ROWS_PER_BENCHMARK,
    allowed_sources: tuple[str, ...] = DEFAULT_ALLOWED_SOURCES,
) -> list[dict[str, object]]:
    return select_proxy_rows(
        json.loads(Path(dataset_results_path).read_text(encoding="utf-8-sig")),
        max_rows_per_benchmark=max_rows_per_benchmark,
        allowed_sources=allowed_sources,
    )


def resolve_seed_placement_def_path(benchmark_root: str | Path, benchmark_name: str) -> Path:
    artifact_paths = resolve_artifact_paths(Path(benchmark_root), benchmark_name)
    candidate = artifact_paths["def_path"].with_name("seed_placed.def")
    if candidate.exists():
        return candidate
    return artifact_paths["def_path"]


def load_phase2_proxy_graphs(
    selected_rows: list[dict[str, object]],
    benchmark_root: str | Path,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
) -> tuple[list[ProblemGraph], torch.Tensor]:
    graphs: list[ProblemGraph] = []
    targets = []
    seed_problem_by_benchmark: dict[str, ChipProblem] = {}
    for row in selected_rows:
        benchmark_name = str(row["benchmark"])
        artifact_paths = resolve_artifact_paths(Path(benchmark_root), benchmark_name)
        placement_def_path = Path(str(row["placement_def"]))
        graph, problem = load_chip_problem(artifact_paths["cells_lef_path"], placement_def_path)
        if benchmark_name not in seed_problem_by_benchmark:
            seed_def_path = resolve_seed_placement_def_path(Path(benchmark_root), benchmark_name)
            _, seed_problem = load_chip_problem(artifact_paths["cells_lef_path"], seed_def_path)
            seed_problem_by_benchmark[benchmark_name] = seed_problem
        graph = add_seed_relative_node_features(
            graph,
            problem,
            seed_problem_by_benchmark[benchmark_name],
        )
        graphs.append(graph)
        targets.append([float(row[name]) for name in target_names])
    return graphs, torch.tensor(targets, dtype=torch.float32)


def load_phase2_proxy_dataset(
    dataset_results_path: str | Path,
    benchmark_root: str | Path,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    max_rows_per_benchmark: int | None = DEFAULT_MAX_ROWS_PER_BENCHMARK,
) -> tuple[list[ProblemGraph], torch.Tensor, list[dict[str, object]]]:
    rows = load_phase2_proxy_rows(
        dataset_results_path,
        max_rows_per_benchmark=max_rows_per_benchmark,
    )
    graphs, targets = load_phase2_proxy_graphs(rows, benchmark_root, target_names=target_names)
    return graphs, targets, rows


def build_phase2_gate(report: dict[str, object]) -> dict[str, object]:
    same_design_mean = float(report.get("same_design_mean_spearman", 0.0))
    cross_design_mean = float(report.get("cross_design_mean_spearman", 0.0))
    return {
        "same_design_threshold": DEFAULT_SAME_DESIGN_SPEARMAN_THRESHOLD,
        "same_design_status": "pass" if same_design_mean >= DEFAULT_SAME_DESIGN_SPEARMAN_THRESHOLD else "fail",
        "cross_design_sanity_threshold": DEFAULT_CROSS_DESIGN_SANITY_THRESHOLD,
        "cross_design_status": "pass" if cross_design_mean > DEFAULT_CROSS_DESIGN_SANITY_THRESHOLD else "fail",
        "cross_design_stretch_threshold": DEFAULT_CROSS_DESIGN_STRETCH_THRESHOLD,
        "cross_design_stretch_status": "pass" if cross_design_mean >= DEFAULT_CROSS_DESIGN_STRETCH_THRESHOLD else "fail",
        "phase3_foundation_ready": same_design_mean >= DEFAULT_SAME_DESIGN_SPEARMAN_THRESHOLD and cross_design_mean > DEFAULT_CROSS_DESIGN_SANITY_THRESHOLD,
    }


def build_proxy_run_plan(
    rows: list[dict[str, object]],
    mode: str = DEFAULT_PROXY_MODE,
    ranking_target: str | None = None,
    benchmark: str | None = None,
    holdout_benchmark: str | None = None,
) -> dict[str, object]:
    same_design_splits = build_same_design_splits(rows, ranking_target=ranking_target)
    if ranking_target is not None:
        same_design_splits = order_same_design_splits(same_design_splits, rows=rows, ranking_target=ranking_target)
    cross_design_splits = build_leave_one_benchmark_out_splits(rows)

    if mode == "same-design":
        if benchmark is not None:
            same_design_splits = [split for split in same_design_splits if str(split["benchmark"]) == benchmark]
        cross_design_splits = []
        run_full_fit = False
    elif mode == "cross-design":
        same_design_splits = []
        if holdout_benchmark is not None:
            cross_design_splits = [split for split in cross_design_splits if str(split["holdout_benchmark"]) == holdout_benchmark]
        run_full_fit = False
    elif mode == "full-fit":
        same_design_splits = []
        cross_design_splits = []
        run_full_fit = True
    elif mode == "all":
        if benchmark is not None:
            same_design_splits = [split for split in same_design_splits if str(split["benchmark"]) == benchmark]
        if holdout_benchmark is not None:
            cross_design_splits = [split for split in cross_design_splits if str(split["holdout_benchmark"]) == holdout_benchmark]
        run_full_fit = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {
        "mode": mode,
        "same_design_splits": same_design_splits,
        "cross_design_splits": cross_design_splits,
        "run_full_fit": run_full_fit,
    }


def run_phase2_proxy(
    dataset_results_path: str | Path,
    benchmark_root: str | Path,
    output_dir: str | Path,
    target_names: tuple[str, ...] = DEFAULT_TARGET_NAMES,
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    max_rows_per_benchmark: int | None = DEFAULT_MAX_ROWS_PER_BENCHMARK,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mode: str = DEFAULT_PROXY_MODE,
    model_family: str = DEFAULT_PROXY_MODEL_FAMILY,
    pooling_mode: str = DEFAULT_PROXY_POOLING_MODE,
    node_path_scale: float = DEFAULT_NODE_PATH_SCALE,
    residual_alpha: float = DEFAULT_FROZEN_BASE_ALPHA,
    benchmark: str | None = None,
    holdout_benchmark: str | None = None,
) -> dict[str, object]:
    selected_rows = load_phase2_proxy_rows(
        dataset_results_path,
        max_rows_per_benchmark=max_rows_per_benchmark,
        allowed_sources=DEFAULT_ALLOWED_SOURCES,
    )
    active_target_names = select_active_target_names(selected_rows, target_names)
    if not active_target_names:
        raise ValueError("Need at least one non-constant target to train the proxy")
    if len(selected_rows) < 2:
        raise ValueError("Need at least two successful dataset rows to train the proxy")
    run_plan = build_proxy_run_plan(
        selected_rows,
        mode=mode,
        ranking_target=active_target_names[0],
        benchmark=benchmark,
        holdout_benchmark=holdout_benchmark,
    )
    if not run_plan["same_design_splits"] and not run_plan["cross_design_splits"] and not run_plan["run_full_fit"]:
        raise ValueError("Proxy run plan is empty")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    folds_dir = output_root / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "phase2_proxy_report.json"
    model_path = output_root / "phase2_proxy.pt"

    report: dict[str, object] = {
        "mode": mode,
        "model_family": model_family,
        "pooling_mode": pooling_mode,
        "node_path_scale": node_path_scale,
        "residual_alpha": residual_alpha,
        "cross_design_folds": [],
        "same_design_folds": [],
        "full_training_row_count": len(_rows_for_run_plan(selected_rows, run_plan)),
        "max_rows_per_benchmark": max_rows_per_benchmark,
        "active_target_names": list(active_target_names),
        "target_signal_summary": summarize_target_signal(selected_rows, active_target_names),
        "mean_openroad_runtime_seconds": sum(float(row["runtime_seconds"]) for row in selected_rows) / len(selected_rows),
        "final_train_loss": None,
    }

    cross_design_reports: list[dict[str, object]] = []
    for split in run_plan["cross_design_splits"]:
        fold_report = _run_proxy_fold(
            rows=selected_rows,
            split=split,
            split_kind="cross-design",
            benchmark_root=benchmark_root,
            target_names=active_target_names,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_family=model_family,
            pooling_mode=pooling_mode,
            node_path_scale=node_path_scale,
            residual_alpha=residual_alpha,
        )
        cross_design_reports.append(fold_report)
        report["cross_design_folds"] = cross_design_reports
        _refresh_aggregate_report(report)
        (folds_dir / f"cross_design__{split['holdout_benchmark']}.json").write_text(json.dumps(fold_report, indent=2))
        report_path.write_text(json.dumps(report, indent=2))

    same_design_reports: list[dict[str, object]] = []
    for split in run_plan["same_design_splits"]:
        fold_report = _run_proxy_fold(
            rows=selected_rows,
            split=split,
            split_kind="same-design",
            benchmark_root=benchmark_root,
            target_names=active_target_names,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            model_family=model_family,
            pooling_mode=pooling_mode,
            node_path_scale=node_path_scale,
            residual_alpha=residual_alpha,
        )
        same_design_reports.append(fold_report)
        report["same_design_folds"] = same_design_reports
        _refresh_aggregate_report(report)
        (folds_dir / f"same_design__{split['benchmark']}.json").write_text(json.dumps(fold_report, indent=2))
        report_path.write_text(json.dumps(report, indent=2))

    final_model = None
    full_schema = None
    if run_plan["run_full_fit"]:
        if model_family == "gnn":
            graphs, targets = load_phase2_proxy_graphs(selected_rows, benchmark_root, target_names=active_target_names)
            full_schema = fit_feature_schema(graphs)
            encoded_graphs = [encode_problem_graph(graph, full_schema) for graph in graphs]
            feature_mean, feature_std = _attach_graph_level_features(
                encoded_graphs,
                selected_rows,
                encoded_graphs,
                selected_rows,
            )
            final_model, final_history = train_proxy_model(
                train_graphs=encoded_graphs,
                train_targets=targets,
                validation_graphs=encoded_graphs,
                validation_targets=targets,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                pooling_mode=pooling_mode,
                node_path_scale=node_path_scale,
            )
            report["full_training_row_count"] = len(graphs)
            report["final_train_loss"] = final_history["final_train_loss"]
            torch.save(
                {
                    "state_dict": final_model.state_dict(),
                    "schema": {
                        "numeric_keys": full_schema.numeric_keys,
                        "categorical_vocab": full_schema.categorical_vocab,
                    },
                    "graph_feature_mean": feature_mean,
                    "graph_feature_std": feature_std,
                    "target_names": active_target_names,
                },
                model_path,
            )
        elif model_family == "graph-stats":
            targets = torch.tensor(
                [[float(row[name]) for name in active_target_names] for row in selected_rows],
                dtype=torch.float32,
            )
            final_model, final_history, _, _ = train_graph_stats_model(
                train_rows=selected_rows,
                train_targets=targets,
                validation_rows=selected_rows,
                validation_targets=targets,
            )
            report["full_training_row_count"] = len(selected_rows)
            report["final_train_loss"] = final_history["final_train_loss"]
            model_path.write_text(
                json.dumps(
                    {
                        "model_family": model_family,
                        "feature_names": list(GRAPH_LEVEL_FEATURE_NAMES),
                        "feature_mean": final_model["feature_mean"].tolist(),
                        "feature_std": final_model["feature_std"].tolist(),
                        "weights": final_model["weights"].tolist(),
                        "target_names": list(active_target_names),
                    },
                    indent=2,
                )
            )
        elif model_family == "frozen-linear-base":
            targets = torch.tensor(
                [[float(row[name]) for name in active_target_names] for row in selected_rows],
                dtype=torch.float32,
            )
            final_model, final_history, _, _, _, _ = train_frozen_linear_base_model(
                train_rows=selected_rows,
                train_targets=targets,
                validation_rows=selected_rows,
                validation_targets=targets,
                benchmark_root=benchmark_root,
                target_names=active_target_names,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                pooling_mode=pooling_mode,
                node_path_scale=node_path_scale,
                residual_alpha=residual_alpha,
            )
            report["full_training_row_count"] = len(selected_rows)
            report["final_train_loss"] = final_history["final_train_loss"]
            model_payload: dict[str, object] = {
                "model_family": model_family,
                "feature_names": list(GRAPH_LEVEL_FEATURE_NAMES),
                "feature_mean": final_model["base_model"]["feature_mean"].tolist(),
                "feature_std": final_model["base_model"]["feature_std"].tolist(),
                "weights": final_model["base_model"]["weights"].tolist(),
                "target_names": list(active_target_names),
                "residual_alpha": residual_alpha,
            }
            if final_model["residual_model"] is not None:
                schema = final_model["schema"]
                model_payload["residual_model"] = {
                    "state_dict": {
                        key: value.detach().cpu().tolist() for key, value in final_model["residual_model"].state_dict().items()
                    },
                    "schema": {
                        "numeric_keys": schema.numeric_keys,
                        "categorical_vocab": schema.categorical_vocab,
                    },
                    "graph_feature_mean": final_model["graph_feature_mean"].tolist(),
                    "graph_feature_std": final_model["graph_feature_std"].tolist(),
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "pooling_mode": pooling_mode,
                    "node_path_scale": node_path_scale,
                }
            model_path.write_text(json.dumps(model_payload, indent=2))
        else:
            raise ValueError(f"Unsupported model_family: {model_family}")

    _refresh_aggregate_report(report)
    report_path.write_text(json.dumps(report, indent=2))
    return {
        "model_path": model_path.resolve().as_posix() if final_model is not None else None,
        "report_path": report_path.resolve().as_posix(),
        "report": report,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 GNN proxy.")
    parser.add_argument(
        "--dataset-results",
        default=str(_repo_root() / "experiments" / "chip" / "phase2_dataset_results.json"),
    )
    parser.add_argument("--benchmark-root", default=str(_repo_root() / "tmp"))
    parser.add_argument("--output-dir", default=str(_repo_root() / "experiments" / "chip" / "phase2_proxy"))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-rows-per-benchmark", type=int, default=DEFAULT_MAX_ROWS_PER_BENCHMARK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--mode", choices=("same-design", "cross-design", "full-fit", "all"), default="same-design")
    parser.add_argument("--model-family", choices=("gnn", "graph-stats", "frozen-linear-base"), default=DEFAULT_PROXY_MODEL_FAMILY)
    parser.add_argument("--pooling-mode", choices=("mean", "sum", "mean-max-std"), default=DEFAULT_PROXY_POOLING_MODE)
    parser.add_argument("--node-path-scale", type=float, default=DEFAULT_NODE_PATH_SCALE)
    parser.add_argument("--residual-alpha", type=float, default=DEFAULT_FROZEN_BASE_ALPHA)
    parser.add_argument("--benchmark")
    parser.add_argument("--holdout-benchmark")
    args = parser.parse_args(argv)

    result = run_phase2_proxy(
        dataset_results_path=args.dataset_results,
        benchmark_root=args.benchmark_root,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_rows_per_benchmark=args.max_rows_per_benchmark,
        batch_size=args.batch_size,
        mode=args.mode,
        model_family=args.model_family,
        pooling_mode=args.pooling_mode,
        node_path_scale=args.node_path_scale,
        residual_alpha=args.residual_alpha,
        benchmark=args.benchmark,
        holdout_benchmark=args.holdout_benchmark,
    )
    print(json.dumps(result, indent=2))


def _iter_batch_indices(length: int, batch_size: int) -> list[list[int]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    return [list(range(start, min(start + batch_size, length))) for start in range(0, length, batch_size)]


def _predict_in_batches(
    model: GNNProxy,
    graphs: list[dict[str, torch.Tensor]],
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for batch_indices in _iter_batch_indices(len(graphs), batch_size):
        graph_batch = [graphs[index] for index in batch_indices]
        outputs.append(model(batch_graph_tensors(graph_batch)))
    return torch.cat(outputs, dim=0)


def _evenly_spaced_take(rows: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count <= 0 or not rows:
        return []
    if count >= len(rows):
        return list(rows)
    if count == 1:
        return [rows[0]]
    positions = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    deduped_positions: list[int] = []
    for position in positions:
        if position not in deduped_positions:
            deduped_positions.append(position)
    while len(deduped_positions) < count:
        for position in range(len(rows)):
            if position not in deduped_positions:
                deduped_positions.append(position)
                if len(deduped_positions) == count:
                    break
    return [rows[position] for position in deduped_positions]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_proxy_fold(
    rows: list[dict[str, object]],
    split: dict[str, object],
    split_kind: str,
    benchmark_root: str | Path,
    target_names: tuple[str, ...],
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    model_family: str,
    pooling_mode: str,
    node_path_scale: float,
    residual_alpha: float,
) -> dict[str, object]:
    train_rows = [rows[index] for index in split["train_indices"]]
    validation_rows = [rows[index] for index in split["validation_indices"]]
    train_targets = torch.tensor([[float(row[name]) for name in target_names] for row in train_rows], dtype=torch.float32)
    validation_targets = torch.tensor(
        [[float(row[name]) for name in target_names] for row in validation_rows],
        dtype=torch.float32,
    )

    start = time.perf_counter()
    if model_family == "gnn":
        train_graphs_raw, _ = load_phase2_proxy_graphs(train_rows, benchmark_root, target_names=target_names)
        validation_graphs_raw, _ = load_phase2_proxy_graphs(validation_rows, benchmark_root, target_names=target_names)
        schema = fit_feature_schema(train_graphs_raw)
        train_graphs = [encode_problem_graph(graph, schema) for graph in train_graphs_raw]
        validation_graphs = [encode_problem_graph(graph, schema) for graph in validation_graphs_raw]
        _attach_graph_level_features(train_graphs, train_rows, validation_graphs, validation_rows)
        model, history = train_proxy_model(
            train_graphs=train_graphs,
            train_targets=train_targets,
            validation_graphs=validation_graphs,
            validation_targets=validation_targets,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            pooling_mode=pooling_mode,
            node_path_scale=node_path_scale,
        )
        with torch.no_grad():
            train_predictions = _predict_in_batches(model, train_graphs, batch_size=batch_size)
            validation_predictions = _predict_in_batches(model, validation_graphs, batch_size=batch_size)
        base_train_predictions = train_predictions
        base_validation_predictions = validation_predictions
    elif model_family == "graph-stats":
        _, history, train_predictions, validation_predictions = train_graph_stats_model(
            train_rows=train_rows,
            train_targets=train_targets,
            validation_rows=validation_rows,
            validation_targets=validation_targets,
        )
        base_train_predictions = train_predictions
        base_validation_predictions = validation_predictions
    elif model_family == "frozen-linear-base":
        (
            _,
            history,
            train_predictions,
            validation_predictions,
            base_train_predictions,
            base_validation_predictions,
        ) = train_frozen_linear_base_model(
            train_rows=train_rows,
            train_targets=train_targets,
            validation_rows=validation_rows,
            validation_targets=validation_targets,
            benchmark_root=benchmark_root,
            target_names=target_names,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            pooling_mode=pooling_mode,
            node_path_scale=node_path_scale,
            residual_alpha=residual_alpha,
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")
    proxy_runtime = time.perf_counter() - start

    fold_report = build_validation_report(
        target_names=target_names,
        y_true=validation_targets,
        y_pred=validation_predictions,
        mean_openroad_runtime_seconds=sum(float(row["runtime_seconds"]) for row in validation_rows) / len(validation_rows),
        mean_proxy_runtime_seconds=proxy_runtime / len(validation_rows),
    )
    train_per_target = compute_per_target_spearman(target_names, train_targets, train_predictions)
    base_per_target = compute_per_target_spearman(target_names, validation_targets, base_validation_predictions)
    fold_report["train_per_target_spearman"] = train_per_target
    fold_report["train_mean_spearman"] = sum(train_per_target.values()) / len(train_per_target)
    fold_report["base_per_target_spearman"] = base_per_target
    fold_report["base_mean_spearman"] = sum(base_per_target.values()) / len(base_per_target)
    fold_report["residual_alpha"] = residual_alpha
    fold_report["train_row_count"] = len(train_rows)
    fold_report["validation_row_count"] = len(validation_rows)
    fold_report["final_train_loss"] = history["final_train_loss"]
    if split_kind == "cross-design":
        fold_report["holdout_benchmark"] = split["holdout_benchmark"]
    elif split_kind == "same-design":
        fold_report["benchmark"] = split["benchmark"]
    else:
        raise ValueError(f"Unsupported split kind: {split_kind}")
    return fold_report


def _refresh_aggregate_report(report: dict[str, object]) -> None:
    cross_design_folds = report.get("cross_design_folds", [])
    same_design_folds = report.get("same_design_folds", [])
    report["cross_design_mean_spearman"] = (
        sum(fold["mean_spearman"] for fold in cross_design_folds) / len(cross_design_folds)
        if cross_design_folds
        else 0.0
    )
    report["cross_design_min_spearman"] = (
        min(fold["mean_spearman"] for fold in cross_design_folds)
        if cross_design_folds
        else 0.0
    )
    report["same_design_mean_spearman"] = (
        sum(fold["mean_spearman"] for fold in same_design_folds) / len(same_design_folds)
        if same_design_folds
        else 0.0
    )
    report["same_design_min_spearman"] = (
        min(fold["mean_spearman"] for fold in same_design_folds)
        if same_design_folds
        else 0.0
    )
    report["phase2_gate"] = build_phase2_gate(report)


def _rows_for_run_plan(
    rows: list[dict[str, object]],
    run_plan: dict[str, object],
) -> list[dict[str, object]]:
    if run_plan["run_full_fit"]:
        return list(rows)
    seen_indices: set[int] = set()
    for split in run_plan["same_design_splits"]:
        seen_indices.update(split["train_indices"])
        seen_indices.update(split["validation_indices"])
    for split in run_plan["cross_design_splits"]:
        seen_indices.update(split["train_indices"])
        seen_indices.update(split["validation_indices"])
    return [rows[index] for index in sorted(seen_indices)]


if __name__ == "__main__":
    main()
