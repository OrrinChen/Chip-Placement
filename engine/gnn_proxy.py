from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBatch:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    graph_index: torch.Tensor
    graph_features: torch.Tensor
    num_graphs: int


def batch_graph_tensors(graphs: list[dict[str, torch.Tensor]]) -> GraphBatch:
    if not graphs:
        raise ValueError("Expected at least one graph")

    node_features: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    graph_indices: list[torch.Tensor] = []
    graph_features: list[torch.Tensor] = []
    node_offset = 0
    feature_dim = graphs[0]["node_features"].shape[1]
    graph_feature_dim = (
        graphs[0]["graph_features"].shape[0]
        if "graph_features" in graphs[0]
        else 0
    )
    for graph_index, graph in enumerate(graphs):
        current_node_features = graph["node_features"].float()
        if current_node_features.ndim != 2:
            raise ValueError("node_features must be rank-2")
        if current_node_features.shape[1] != feature_dim:
            raise ValueError("all graphs must share the same node feature dimension")
        current_edge_index = graph["edge_index"].long()
        if current_edge_index.ndim != 2 or current_edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")
        node_count = current_node_features.shape[0]
        node_features.append(current_node_features)
        graph_indices.append(torch.full((node_count,), graph_index, dtype=torch.long))
        current_graph_features = graph.get("graph_features")
        if current_graph_features is None:
            current_graph_features = torch.zeros((graph_feature_dim,), dtype=torch.float32)
        current_graph_features = current_graph_features.float().reshape(-1)
        if current_graph_features.shape[0] != graph_feature_dim:
            raise ValueError("all graphs must share the same graph feature dimension")
        graph_features.append(current_graph_features)
        if current_edge_index.numel():
            edge_indices.append(current_edge_index + node_offset)
        node_offset += node_count

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return GraphBatch(
        node_features=torch.cat(node_features, dim=0),
        edge_index=edge_index,
        graph_index=torch.cat(graph_indices, dim=0),
        graph_features=torch.stack(graph_features, dim=0),
        num_graphs=len(graphs),
    )


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_count = node_features.shape[0]
        device = node_features.device
        aggregated = torch.zeros((node_count, node_features.shape[1]), device=device, dtype=node_features.dtype)
        counts = torch.zeros((node_count, 1), device=device, dtype=node_features.dtype)
        if edge_index.numel():
            source, target = edge_index
            aggregated.index_add_(0, target, node_features[source])
            counts.index_add_(0, target, torch.ones((target.shape[0], 1), device=device, dtype=node_features.dtype))
        mean_neighbors = aggregated / counts.clamp_min(1.0)
        return torch.relu(self.self_linear(node_features) + self.neighbor_linear(mean_neighbors))


class GNNProxy(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        graph_feature_dim: int = 0,
        pooling_mode: str = "mean",
        node_path_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if pooling_mode not in {"mean", "sum", "mean-max-std"}:
            raise ValueError("pooling_mode must be one of: mean, sum, mean-max-std")

        layers: list[nn.Module] = []
        input_dim = node_feature_dim
        for _ in range(num_layers):
            layers.append(GraphSAGELayer(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.graph_feature_dim = graph_feature_dim
        self.pooling_mode = pooling_mode
        self.node_path_scale = node_path_scale
        pooled_dim = hidden_dim if pooling_mode in {"mean", "sum"} else hidden_dim * 3
        self.head = nn.Sequential(
            nn.Linear(pooled_dim + graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        node_embeddings = batch.node_features
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, batch.edge_index)
        pooled = self._pool(node_embeddings, batch.graph_index, batch.num_graphs)
        if self.node_path_scale != 1.0:
            pooled = pooled * self.node_path_scale
        if batch.graph_features.numel():
            pooled = torch.cat([pooled, batch.graph_features], dim=1)
        return self.head(pooled)

    def _pool(
        self,
        node_embeddings: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.pooling_mode == "mean":
            return self._mean_pool(node_embeddings, graph_index, num_graphs)
        if self.pooling_mode == "sum":
            return self._sum_pool(node_embeddings, graph_index, num_graphs)
        mean_pooled = self._mean_pool(node_embeddings, graph_index, num_graphs)
        max_pooled = self._max_pool(node_embeddings, graph_index, num_graphs)
        std_pooled = self._std_pool(node_embeddings, graph_index, num_graphs, mean_pooled)
        return torch.cat([mean_pooled, max_pooled, std_pooled], dim=1)

    def _sum_pool(
        self,
        node_embeddings: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        device = node_embeddings.device
        pooled = torch.zeros((num_graphs, node_embeddings.shape[1]), device=device, dtype=node_embeddings.dtype)
        pooled.index_add_(0, graph_index, node_embeddings)
        return pooled

    def _mean_pool(
        self,
        node_embeddings: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled = self._sum_pool(node_embeddings, graph_index, num_graphs)
        device = node_embeddings.device
        counts = torch.zeros((num_graphs, 1), device=device, dtype=node_embeddings.dtype)
        counts.index_add_(0, graph_index, torch.ones((graph_index.shape[0], 1), device=device, dtype=node_embeddings.dtype))
        return pooled / counts.clamp_min(1.0)

    def _max_pool(
        self,
        node_embeddings: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        pooled = []
        for graph_id in range(num_graphs):
            mask = graph_index == graph_id
            pooled.append(node_embeddings[mask].max(dim=0).values if mask.any() else torch.zeros_like(node_embeddings[0]))
        return torch.stack(pooled, dim=0)

    def _std_pool(
        self,
        node_embeddings: torch.Tensor,
        graph_index: torch.Tensor,
        num_graphs: int,
        mean_pooled: torch.Tensor,
    ) -> torch.Tensor:
        device = node_embeddings.device
        squared_diffs = torch.zeros((num_graphs, node_embeddings.shape[1]), device=device, dtype=node_embeddings.dtype)
        counts = torch.zeros((num_graphs, 1), device=device, dtype=node_embeddings.dtype)
        centered = node_embeddings - mean_pooled[graph_index]
        squared_diffs.index_add_(0, graph_index, centered.square())
        counts.index_add_(0, graph_index, torch.ones((graph_index.shape[0], 1), device=device, dtype=node_embeddings.dtype))
        variance = squared_diffs / counts.clamp_min(1.0)
        return torch.sqrt(variance.clamp_min(0.0))
