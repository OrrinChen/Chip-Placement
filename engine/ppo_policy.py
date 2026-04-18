from __future__ import annotations

import torch
from torch import nn

from engine.gnn_proxy import GraphBatch
from engine.gnn_proxy import GraphSAGELayer


class PPOPlacementPolicy(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        graph_feature_dim: int,
        grid_channels: int,
        grid_shape: tuple[int, int],
        hidden_dim: int = 64,
        num_graph_layers: int = 2,
        max_macro_index: int = 512,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.hidden_dim = hidden_dim
        self.graph_feature_dim = graph_feature_dim

        layers: list[nn.Module] = []
        input_dim = node_feature_dim
        for _ in range(num_graph_layers):
            layers.append(GraphSAGELayer(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.graph_layers = nn.ModuleList(layers)

        self.current_macro_embedding = nn.Embedding(max_macro_index, hidden_dim)
        self.graph_feature_proj = nn.Linear(graph_feature_dim, hidden_dim) if graph_feature_dim > 0 else None

        self.canvas_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.film = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        self.policy_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("value_target_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("value_target_std", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        *,
        graph_batch: GraphBatch,
        canvas: torch.Tensor,
        current_macro_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_logits, value_train_space = self.forward_train_space(
            graph_batch=graph_batch,
            canvas=canvas,
            current_macro_index=current_macro_index,
            action_mask=action_mask,
        )
        return masked_logits, self.decode_value_predictions(value_train_space)

    def forward_train_space(
        self,
        *,
        graph_batch: GraphBatch,
        canvas: torch.Tensor,
        current_macro_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_embeddings = graph_batch.node_features
        for layer in self.graph_layers:
            node_embeddings = layer(node_embeddings, graph_batch.edge_index)

        global_graph_embedding = self._mean_pool(node_embeddings, graph_batch.graph_index, graph_batch.num_graphs)
        if self.graph_feature_proj is not None and graph_batch.graph_features.numel():
            global_graph_embedding = global_graph_embedding + self.graph_feature_proj(graph_batch.graph_features)

        current_macro_embedding = self.current_macro_embedding(current_macro_index)
        conditioning = torch.cat([global_graph_embedding, current_macro_embedding], dim=1)
        film_scale, film_bias = self.film(conditioning).chunk(2, dim=1)

        canvas_features = self.canvas_encoder(canvas)
        film_scale = film_scale.unsqueeze(-1).unsqueeze(-1)
        film_bias = film_bias.unsqueeze(-1).unsqueeze(-1)
        conditioned_canvas = canvas_features * (1.0 + film_scale) + film_bias

        logits = self.policy_head(conditioned_canvas).flatten(start_dim=1)
        masked_logits = logits.masked_fill(action_mask.view(action_mask.shape[0], -1), float("-inf"))

        pooled_canvas = conditioned_canvas.mean(dim=(2, 3))
        value_features = torch.cat([global_graph_embedding, current_macro_embedding, pooled_canvas], dim=1)
        value_train_space = self.value_head(value_features)
        return masked_logits, value_train_space

    def set_value_target_stats(self, *, mean: float, std: float) -> None:
        safe_std = float(std) if abs(float(std)) > 1e-8 else 1.0
        self.value_target_mean.copy_(
            torch.tensor(float(mean), dtype=self.value_target_mean.dtype, device=self.value_target_mean.device)
        )
        self.value_target_std.copy_(
            torch.tensor(safe_std, dtype=self.value_target_std.dtype, device=self.value_target_std.device)
        )

    def encode_value_targets(self, raw_returns: torch.Tensor) -> torch.Tensor:
        mean = self.value_target_mean.to(device=raw_returns.device, dtype=raw_returns.dtype)
        std = self.value_target_std.to(device=raw_returns.device, dtype=raw_returns.dtype)
        return (raw_returns - mean) / std

    def decode_value_predictions(self, value_prediction_train_space: torch.Tensor) -> torch.Tensor:
        mean = self.value_target_mean.to(
            device=value_prediction_train_space.device,
            dtype=value_prediction_train_space.dtype,
        )
        std = self.value_target_std.to(
            device=value_prediction_train_space.device,
            dtype=value_prediction_train_space.dtype,
        )
        return (value_prediction_train_space * std) + mean

    def predict_value_train_space(
        self,
        *,
        graph_batch: GraphBatch,
        canvas: torch.Tensor,
        current_macro_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, value_train_space = self.forward_train_space(
            graph_batch=graph_batch,
            canvas=canvas,
            current_macro_index=current_macro_index,
            action_mask=action_mask,
        )
        return value_train_space

    def predict_value_raw(
        self,
        *,
        graph_batch: GraphBatch,
        canvas: torch.Tensor,
        current_macro_index: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        value_train_space = self.predict_value_train_space(
            graph_batch=graph_batch,
            canvas=canvas,
            current_macro_index=current_macro_index,
            action_mask=action_mask,
        )
        return self.decode_value_predictions(value_train_space)

    @staticmethod
    def _mean_pool(node_embeddings: torch.Tensor, graph_index: torch.Tensor, num_graphs: int) -> torch.Tensor:
        device = node_embeddings.device
        pooled = torch.zeros((num_graphs, node_embeddings.shape[1]), device=device, dtype=node_embeddings.dtype)
        counts = torch.zeros((num_graphs, 1), device=device, dtype=node_embeddings.dtype)
        pooled.index_add_(0, graph_index, node_embeddings)
        counts.index_add_(0, graph_index, torch.ones((graph_index.shape[0], 1), device=device, dtype=node_embeddings.dtype))
        return pooled / counts.clamp_min(1.0)
