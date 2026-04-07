from __future__ import annotations

import torch

from chip.experiments.phase3_imitation import masked_behavior_cloning_loss


def test_masked_behavior_cloning_ignores_invalid_actions():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    action_index = torch.tensor([1], dtype=torch.int64)
    invalid_action_mask = torch.tensor([[False, False, True, True]])

    loss = masked_behavior_cloning_loss(logits, action_index, invalid_action_mask)

    assert loss.item() > 0.0
    assert torch.isfinite(loss)
