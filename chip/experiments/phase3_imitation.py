from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_behavior_cloning_loss(
    logits: torch.Tensor,
    action_index: torch.Tensor,
    invalid_action_mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(invalid_action_mask, float("-inf"))
    return F.cross_entropy(masked_logits, action_index)
