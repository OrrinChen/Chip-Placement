from __future__ import annotations

import torch

from engine.gnn_proxy import batch_graph_tensors
from engine.ppo_policy import PPOPlacementPolicy


def build_batch():
    graphs = [
        {
            "node_features": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.5, 0.5, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "graph_features": torch.tensor([0.1, 0.2], dtype=torch.float32),
        },
        {
            "node_features": torch.tensor(
                [
                    [0.2, 0.8, 0.0, 1.0],
                    [0.9, 0.1, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "graph_features": torch.tensor([0.3, 0.4], dtype=torch.float32),
        },
    ]
    return batch_graph_tensors(graphs)


def test_ppo_policy_outputs_logits_and_value():
    batch = build_batch()
    canvas = torch.randn(2, 6, 4, 4)
    current_macro_index = torch.tensor([0, 1], dtype=torch.long)
    action_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    model = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=2,
        grid_channels=6,
        grid_shape=(4, 4),
        hidden_dim=16,
        max_macro_index=8,
    )

    logits, value = model(
        graph_batch=batch,
        canvas=canvas,
        current_macro_index=current_macro_index,
        action_mask=action_mask,
    )

    assert tuple(logits.shape) == (2, 16)
    assert tuple(value.shape) == (2, 1)


def test_ppo_policy_masks_invalid_actions_with_negative_infinity():
    batch = build_batch()
    canvas = torch.randn(2, 6, 4, 4)
    current_macro_index = torch.tensor([0, 1], dtype=torch.long)
    action_mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    action_mask[0, 0, 0] = True
    model = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=2,
        grid_channels=6,
        grid_shape=(4, 4),
        hidden_dim=16,
        max_macro_index=8,
    )

    logits, _ = model(
        graph_batch=batch,
        canvas=canvas,
        current_macro_index=current_macro_index,
        action_mask=action_mask,
    )

    assert torch.isneginf(logits[0, 0])
    assert torch.isfinite(logits[0, 1:]).all()
