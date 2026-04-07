from __future__ import annotations

from pathlib import Path

import torch

from chip.experiments.phase3_ppo import load_policy_warm_start
from engine.ppo_policy import PPOPlacementPolicy


def test_load_policy_warm_start_ignores_missing_value_head(tmp_path: Path):
    policy = PPOPlacementPolicy(
        node_feature_dim=4,
        graph_feature_dim=0,
        grid_channels=6,
        grid_shape=(4, 4),
    )
    checkpoint = {
        "policy_state_dict": {
            key: value
            for key, value in policy.state_dict().items()
            if not key.startswith("value_head.")
        }
    }
    path = tmp_path / "bc.pt"
    torch.save(checkpoint, path)

    loaded = load_policy_warm_start(policy, path)

    assert loaded is True
