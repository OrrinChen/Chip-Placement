from __future__ import annotations

from pathlib import Path

import torch


def load_policy_warm_start(policy, checkpoint_path: Path) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["policy_state_dict"]
    policy.load_state_dict(state_dict, strict=False)
    return True
