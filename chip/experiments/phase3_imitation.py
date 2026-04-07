from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from chip.imitation import DemonstrationStep
from chip.imitation import observation_from_demonstration
from chip.problem import ChipProblem
from engine.gnn_proxy import GraphBatch
from engine.gnn_proxy import batch_graph_tensors
from engine.graph import ProblemGraph
from engine.ppo_policy import PPOPlacementPolicy


@dataclass(frozen=True)
class FeatureSchema:
    numeric_keys: tuple[str, ...]
    categorical_vocab: dict[str, tuple[str, ...]]


def masked_behavior_cloning_loss(
    logits: torch.Tensor,
    action_index: torch.Tensor,
    invalid_action_mask: torch.Tensor,
) -> torch.Tensor:
    if invalid_action_mask.ndim > 2:
        invalid_action_mask = invalid_action_mask.view(invalid_action_mask.shape[0], -1)
    masked_logits = logits.masked_fill(invalid_action_mask, float("-inf"))
    return F.cross_entropy(masked_logits, action_index)


def train_behavior_cloning_policy(
    *,
    problem: ChipProblem,
    demonstrations: list[DemonstrationStep],
    grid_shape: tuple[int, int] = (32, 32),
    hidden_dim: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    if not demonstrations:
        raise ValueError("Expected at least one demonstration")

    torch.manual_seed(0)
    train_data = build_behavior_cloning_batch(problem=problem, demonstrations=demonstrations, grid_shape=grid_shape)
    policy = PPOPlacementPolicy(
        node_feature_dim=train_data.graph_batch.node_features.shape[1],
        graph_feature_dim=train_data.graph_batch.graph_features.shape[1],
        grid_channels=train_data.canvas.shape[1],
        grid_shape=grid_shape,
        hidden_dim=hidden_dim,
        max_macro_index=max(int(train_data.current_macro_index.max().item()) + 2, 8),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    final_loss = None
    for _ in range(epochs):
        logits, _ = policy(
            graph_batch=train_data.graph_batch,
            canvas=train_data.canvas,
            current_macro_index=train_data.current_macro_index,
            action_mask=train_data.action_mask,
        )
        loss = masked_behavior_cloning_loss(logits, train_data.action_index, train_data.action_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss

    evaluation = evaluate_behavior_cloning_policy(
        policy=policy,
        problem=problem,
        demonstrations=demonstrations,
        grid_shape=grid_shape,
    )
    if checkpoint_path is not None:
        export_behavior_cloning_checkpoint(policy, checkpoint_path)

    return {
        "policy": policy,
        "masked_accuracy": evaluation["masked_accuracy"],
        "final_loss": float(final_loss.item()) if final_loss is not None else 0.0,
        "schema": train_data.schema,
    }


def evaluate_behavior_cloning_policy(
    *,
    policy: PPOPlacementPolicy,
    problem: ChipProblem,
    demonstrations: list[DemonstrationStep],
    grid_shape: tuple[int, int] = (32, 32),
) -> dict[str, float]:
    evaluation_data = build_behavior_cloning_batch(problem=problem, demonstrations=demonstrations, grid_shape=grid_shape)
    with torch.no_grad():
        logits, _ = policy(
            graph_batch=evaluation_data.graph_batch,
            canvas=evaluation_data.canvas,
            current_macro_index=evaluation_data.current_macro_index,
            action_mask=evaluation_data.action_mask,
        )
    predicted = logits.argmax(dim=1)
    accuracy = (predicted == evaluation_data.action_index).float().mean().item()
    loss = masked_behavior_cloning_loss(logits, evaluation_data.action_index, evaluation_data.action_mask).item()
    return {"masked_accuracy": float(accuracy), "loss": float(loss)}


@dataclass(frozen=True)
class BehaviorCloningBatch:
    graph_batch: GraphBatch
    canvas: torch.Tensor
    current_macro_index: torch.Tensor
    action_mask: torch.Tensor
    action_index: torch.Tensor
    schema: FeatureSchema


def build_behavior_cloning_batch(
    *,
    problem: ChipProblem,
    demonstrations: list[DemonstrationStep],
    grid_shape: tuple[int, int] = (32, 32),
) -> BehaviorCloningBatch:
    observations = [
        observation_from_demonstration(problem=problem, demonstration=demonstration, grid_shape=grid_shape)
        for demonstration in demonstrations
    ]
    schema = fit_feature_schema([observation["graph"] for observation in observations])
    graph_batch = batch_graph_tensors([encode_problem_graph(observation["graph"], schema) for observation in observations])
    return BehaviorCloningBatch(
        graph_batch=graph_batch,
        canvas=torch.stack([observation["canvas"].float() for observation in observations], dim=0),
        current_macro_index=torch.tensor(
            [int(observation["current_macro_index"]) for observation in observations],
            dtype=torch.long,
        ),
        action_mask=torch.stack([observation["action_mask"].bool() for observation in observations], dim=0),
        action_index=torch.tensor([int(demonstration.action_index) for demonstration in demonstrations], dtype=torch.long),
        schema=schema,
    )


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

    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = [(index_by_node[source], index_by_node[target]) for source, target, _ in graph.edges]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return {
        "node_features": torch.tensor(node_features, dtype=torch.float32),
        "edge_index": edge_index,
        "graph_features": torch.empty((0,), dtype=torch.float32),
    }


def export_behavior_cloning_checkpoint(policy: PPOPlacementPolicy, checkpoint_path: Path) -> None:
    filtered_state_dict = {
        key: value.detach().cpu()
        for key, value in policy.state_dict().items()
        if not key.startswith("value_head.")
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": filtered_state_dict}, checkpoint_path)
