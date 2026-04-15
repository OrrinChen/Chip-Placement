from __future__ import annotations

from pathlib import Path

from chip.experiments.phase3_macroplacement_bringup import build_ariane_formulation_config
from chip.experiments.phase3_macroplacement_bringup import build_ariane_reward_model
from chip.experiments.phase3_macroplacement_bringup import summarize_formulation_sweep


def test_build_ariane_formulation_config_promotes_64_grid():
    config = build_ariane_formulation_config()

    assert config["grid_shape"] == (64, 64)
    assert config["macro_order_strategy"] == "descending_area"


def test_build_ariane_reward_model_disables_phase2_proxy_weight_for_bringup(tmp_path: Path):
    reward_model, metadata = build_ariane_reward_model(artifact_dir=tmp_path)

    assert reward_model.weights.w_proxy == 0.0
    assert reward_model.weights.w_hpwl == 1.0
    assert reward_model.weights.completion_bonus == 0.0
    assert reward_model.normalize_rewards is False
    assert metadata["reward_weights"]["w_proxy"] == 0.0
    assert metadata["reward_weights"]["w_hpwl"] == 1.0
    assert metadata["reward_weights"]["completion_bonus"] == 0.0


def test_summarize_formulation_sweep_identifies_best_hpwl_and_completion_gain():
    summary = summarize_formulation_sweep(
        runs=[
            {
                "label": "g32_desc",
                "grid_shape": [32, 32],
                "macro_order_strategy": "descending_area",
                "episode": {"final_hpwl": 10.0, "steps": 90, "completed_episode": False, "termination_reason": "no_valid_actions"},
            },
            {
                "label": "g64_seedyx",
                "grid_shape": [64, 64],
                "macro_order_strategy": "seed_yx",
                "episode": {"final_hpwl": 7.0, "steps": 120, "completed_episode": True, "termination_reason": "completed"},
            },
        ],
        canonical_label="g32_desc",
    )

    assert summary["best_hpwl"]["label"] == "g64_seedyx"
    assert summary["best_steps"]["label"] == "g64_seedyx"
    assert summary["canonical_comparison"]["hpwl_delta"] == -3.0
    assert summary["canonical_comparison"]["completion_improved"] is True
