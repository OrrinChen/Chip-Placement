from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chip.experiments import phase3_macroplacement_ppo


class _StubAdapter:
    def ordered_macros(self):
        return ("m1", "m2")


def test_run_ariane_ppo_checkpoint_records_gap_progress(monkeypatch, tmp_path: Path):
    captured_train_kwargs = {}
    manual_seed_calls = []
    warm_start_calls = []

    def fake_manual_seed(value):
        manual_seed_calls.append(value)

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 2,
            "ordered_macro_count": 2,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0, "legality_penalty": 10.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), object(), reward_model, _StubAdapter(), metadata

    def fake_train_policy(**kwargs):
        captured_train_kwargs.update(kwargs)
        return {
            "history": [
                {
                    "final_hpwl": 90.0,
                    "best_hpwl_so_far": 90.0,
                    "steps": 2.0,
                    "policy_loss": 1.0,
                    "value_loss": 2.0,
                    "entropy": 1.5,
                    "invalid_action_count": 0.0,
                    "episode_return": -90.0,
                    "completed_episode": 0.0,
                    "is_legal": 1.0,
                    "approx_kl": 0.01,
                },
                {
                    "final_hpwl": 68.0,
                    "best_hpwl_so_far": 68.0,
                    "steps": 2.0,
                    "policy_loss": 0.5,
                    "value_loss": 1.0,
                    "entropy": 1.0,
                    "invalid_action_count": 0.0,
                    "episode_return": -68.0,
                    "completed_episode": 1.0,
                    "is_legal": 1.0,
                    "approx_kl": 0.01,
                },
            ],
                "final_rollout": {
                    "steps": 2,
                    "episode_return": -68.0,
                    "final_hpwl": 68.0,
                    "invalid_action_count": 0,
                    "termination_reason": "completed",
                    "completed_episode": True,
                "is_legal": True,
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: warm_start_calls.append((args, kwargs)) or True)
    monkeypatch.setattr(phase3_macroplacement_ppo.torch, "manual_seed", fake_manual_seed)

    random_artifact = tmp_path / "phase3_masked_random_ariane.json"
    random_artifact.write_text('{"summary":{"final_hpwl":{"mean":100.0}}}', encoding="utf-8")
    greedy_artifact = tmp_path / "phase3_greedy_ariane.json"
    greedy_artifact.write_text('{"episode":{"final_hpwl":60.0}}', encoding="utf-8")

    artifact = phase3_macroplacement_ppo.run_ariane_ppo_checkpoint(
        output_path=tmp_path / "phase3_ppo_ariane_checkpoint.json",
        num_updates=2,
        )

    assert artifact["metadata"]["grid_shape"] == [64, 64]
    assert artifact["history"][0]["gap_to_greedy"] == 30.0
    assert artifact["history"][1]["gap_to_greedy"] == 8.0
    assert artifact["checkpoint_summary"]["best_gap_to_greedy"] == 8.0
    assert artifact["checkpoint_summary"]["first_update_reaching_80pct_of_greedy_gap"] == 2
    assert artifact["checkpoint_summary"]["first_update_reaching_beam_level"] is None
    assert artifact["checkpoint_summary"]["completed_episode_rate"] == 0.5
    assert artifact["checkpoint_summary"]["remaining_macro_count"] == 0
    assert artifact["history_step_histogram"] == {"2": 2}
    assert artifact["reference_levels_status"]["available"] is True
    assert artifact["checkpoint_gate"]["reference_levels_available"] is True
    assert artifact["metadata"]["training_config"]["value_coef"] == 0.0
    assert artifact["metadata"]["training_config"]["entropy_coef"] == 5.0
    assert artifact["metadata"]["training_config"]["learning_rate"] == 1e-4
    assert artifact["metadata"]["training_config"]["kl_anchor_beta"] == 5.0
    assert artifact["metadata"]["training_config"]["init_seed"] == 0
    assert captured_train_kwargs["value_coef"] == 0.0
    assert captured_train_kwargs["entropy_coef"] == 5.0
    assert captured_train_kwargs["learning_rate"] == 1e-4
    assert captured_train_kwargs["kl_anchor_beta"] == 5.0
    assert captured_train_kwargs["reference_policy"] is not None
    assert manual_seed_calls == [0]
    assert len(warm_start_calls) == 1


def test_run_ariane_b2_separated_compare_uses_distinct_scratch_and_warm_configs(monkeypatch, tmp_path: Path):
    train_calls = []
    prefit_calls = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 2,
            "ordered_macro_count": 2,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0, "legality_penalty": 10.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), object(), reward_model, _StubAdapter(), metadata

    def fake_train_policy(**kwargs):
        train_calls.append(kwargs)
        final_hpwl = 90.0 - (5.0 * len(train_calls))
        return {
            "history": [
                {
                    "final_hpwl": final_hpwl,
                    "best_hpwl_so_far": final_hpwl,
                    "policy_loss": 1.0,
                    "value_loss": 2.0,
                    "entropy": 1.5,
                    "invalid_action_count": 0.0,
                    "episode_return": -final_hpwl,
                    "completed_episode": 0.0,
                    "is_legal": 1.0,
                    "approx_kl": 0.01,
                }
            ],
            "final_rollout": {
                "steps": 2,
                "episode_return": -final_hpwl,
                "final_hpwl": final_hpwl,
                "invalid_action_count": 0,
                "termination_reason": "completed",
                "completed_episode": True,
                "is_legal": True,
            },
        }

    def fake_prefit_value_head_from_rollout(**kwargs):
        prefit_calls.append(kwargs)
        return {"initial_loss": 4.0, "final_loss": 1.0, "epochs": 5}

    def fake_collect_rollout(**kwargs):
        return {
            "steps": 2,
            "episode_return": -60.0,
            "final_hpwl": 60.0,
            "invalid_action_count": 0,
            "termination_reason": "completed",
            "completed_episode": True,
            "is_legal": True,
            "transitions": [],
            "schema": None,
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "initialize_policy_for_problem", lambda **kwargs: object())
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollout", fake_prefit_value_head_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)

    random_artifact = tmp_path / "phase3_masked_random_ariane.json"
    random_artifact.write_text('{"summary":{"final_hpwl":{"mean":100.0}}}', encoding="utf-8")
    greedy_artifact = tmp_path / "phase3_greedy_ariane.json"
    greedy_artifact.write_text('{"episode":{"final_hpwl":60.0}}', encoding="utf-8")

    artifact = phase3_macroplacement_ppo.run_ariane_b2_separated_compare(
        output_path=tmp_path / "phase3_ariane_b2_compare.json",
        num_updates=5,
    )

    assert artifact["runs"]["scratch"]["training_config"]["learning_rate"] == 1e-3
    assert artifact["runs"]["warm_entropy"]["training_config"]["learning_rate"] == 1e-4
    assert artifact["runs"]["warm_entropy"]["training_config"]["entropy_coef"] == 5.0
    assert artifact["runs"]["warm_value_prefit"]["training_config"]["learning_rate"] == 1e-4
    assert artifact["runs"]["warm_value_prefit"]["prefit"]["final_loss"] == 1.0
    assert len(prefit_calls) == 1
    assert len(train_calls) == 3


def test_load_ariane_reference_levels_falls_back_to_parent_directory(tmp_path: Path):
    artifact_dir = tmp_path / "child"
    artifact_dir.mkdir()
    (tmp_path / "phase3_masked_random_ariane.json").write_text('{"summary":{"final_hpwl":{"mean":100.0}}}', encoding="utf-8")
    (tmp_path / "phase3_greedy_ariane.json").write_text('{"episode":{"final_hpwl":60.0}}', encoding="utf-8")

    random_mean_hpwl, greedy_final_hpwl, warnings = phase3_macroplacement_ppo._load_ariane_reference_levels(artifact_dir)

    assert random_mean_hpwl == 100.0
    assert greedy_final_hpwl == 60.0
    assert warnings == []


def test_load_ariane_reference_levels_reports_missing_references(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(phase3_macroplacement_ppo, "_repo_root", lambda: tmp_path)
    random_mean_hpwl, greedy_final_hpwl, warnings = phase3_macroplacement_ppo._load_ariane_reference_levels(tmp_path)

    assert random_mean_hpwl is None
    assert greedy_final_hpwl is None
    assert warnings == ["missing_masked_random_reference", "missing_greedy_reference"]


def test_resolve_ariane_warm_start_checkpoint_falls_back_to_canonical_pilot(monkeypatch, tmp_path: Path):
    fallback_dir = tmp_path / "experiments" / "chip" / "phase3_macroplacement_warmstart_pilot_v2"
    fallback_dir.mkdir(parents=True)
    fallback = fallback_dir / "ariane_greedy_bc.pt"
    fallback.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(phase3_macroplacement_ppo, "_repo_root", lambda: tmp_path)

    resolved = phase3_macroplacement_ppo.resolve_ariane_warm_start_checkpoint(tmp_path / "missing_artifact_dir")

    assert resolved == fallback


def test_summarize_termination_audit_reports_remaining_macros_and_failure_macro():
    summary = phase3_macroplacement_ppo.summarize_termination_audit(
        ordered_macros=("m0", "m1", "m2", "m3"),
        trace=[
            {"step_index": 0, "macro": "m0", "target": (1.0, 1.0), "legal_action_count": 12},
            {"step_index": 1, "macro": "m1", "target": (2.0, 2.0), "legal_action_count": 5},
        ],
        termination_reason="no_valid_actions",
        final_hpwl=123.0,
        episode_return=-10.0,
        is_legal=True,
    )

    assert summary["placed_macro_count"] == 2
    assert summary["remaining_macro_count"] == 2
    assert summary["failing_macro"] == "m2"
    assert summary["last_step"]["legal_action_count"] == 5
    assert summary["completed_episode"] is False


def test_find_first_trace_divergence_returns_first_mismatch():
    divergence = phase3_macroplacement_ppo.find_first_trace_divergence(
        reference_trace=[
            {"step_index": 0, "macro": "m0", "target": (1.0, 1.0)},
            {"step_index": 1, "macro": "m1", "target": (2.0, 2.0)},
        ],
        candidate_trace=[
            {"step_index": 0, "macro": "m0", "target": (1.0, 1.0)},
            {"step_index": 1, "macro": "m1", "target": (3.0, 2.0)},
        ],
    )

    assert divergence == {
        "step_index": 1,
        "reference": {"step_index": 1, "macro": "m1", "target": [2.0, 2.0]},
        "candidate": {"step_index": 1, "macro": "m1", "target": [3.0, 2.0]},
    }


def test_run_ariane_termination_audit_records_failure_and_divergence(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 3,
            "ordered_macro_count": 3,
            "reward_metadata": {},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), object(), reward_model, _StubAdapter(), metadata

    def fake_train_policy(**kwargs):
        return {
            "history": [],
            "final_rollout": {
                "steps": 2,
                "episode_return": -80.0,
                "final_hpwl": 80.0,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    def fake_collect_greedy(*args, **kwargs):
        return {
            "trace": [
                {"step_index": 0, "macro": "m1", "target": (1.0, 1.0), "legal_action_count": 10},
                {"step_index": 1, "macro": "m2", "target": (2.0, 2.0), "legal_action_count": 8},
            ],
            "steps": 2,
            "placed_macro_count": 2,
            "remaining_macro_count": 0,
            "termination_reason": "completed",
            "final_hpwl": 60.0,
            "episode_return": -60.0,
            "is_legal": True,
            "completed_episode": True,
        }

    def fake_collect_policy(*args, **kwargs):
        return {
            "trace": [
                {"step_index": 0, "macro": "m1", "target": (1.0, 1.0), "legal_action_count": 10},
                {"step_index": 1, "macro": "m2", "target": (5.0, 5.0), "legal_action_count": 1},
            ],
            "steps": 2,
            "placed_macro_count": 2,
            "remaining_macro_count": 0,
            "termination_reason": "no_valid_actions",
            "final_hpwl": 80.0,
            "episode_return": -80.0,
            "is_legal": True,
            "completed_episode": False,
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_collect_greedy_episode_audit", fake_collect_greedy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_collect_policy_rollout_audit", fake_collect_policy)

    artifact = phase3_macroplacement_ppo.run_ariane_termination_audit(
        output_path=tmp_path / "phase3_ariane_termination_audit.json",
        num_updates=5,
    )

    assert artifact["ppo"]["termination_reason"] == "no_valid_actions"
    assert artifact["ppo"]["remaining_macro_count"] == 0
    assert artifact["greedy"]["completed_episode"] is True


def test_run_ariane_critic_prefit_comparison_records_baseline_fair_shot_and_zscore(monkeypatch, tmp_path: Path):
    prefit_calls = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_collect_policy_rollout(**kwargs):
        seed = int(kwargs["seed"])
        base_hpwl = 300.0 - (10.0 * seed)
        return {
            "steps": 2,
            "episode_return": -base_hpwl,
            "final_hpwl": base_hpwl,
            "invalid_action_count": 0,
            "termination_reason": "no_valid_actions",
            "completed_episode": False,
            "is_legal": True,
            "schema": SimpleNamespace(name="feature_schema"),
            "transitions": [
                SimpleNamespace(
                    observation={"step_index": 0, "hpwl": base_hpwl + 20.0},
                    action_index=0,
                    log_prob=0.0,
                    reward=5.0,
                    value=0.0,
                ),
                SimpleNamespace(
                    observation={"step_index": 1, "hpwl": base_hpwl},
                    action_index=1,
                    log_prob=0.0,
                    reward=10.0,
                    value=0.0,
                ),
            ],
        }

    def fake_prefit_value_head_from_rollouts(**kwargs):
        prefit_calls.append(kwargs)
        transform = kwargs["target_transform"]
        correlation = {"raw": 0.1, "zscore": 0.8}[transform]
        if kwargs["epochs"] > 100 and transform == "raw":
            correlation = 0.2
        return {
            "initial_loss": 10.0,
            "final_loss": 1.0,
            "epochs": float(kwargs["epochs"]),
            "rollout_count": float(len(kwargs["fixed_rollouts"])),
            "transition_count": float(sum(len(rollout.transitions) for rollout in kwargs["fixed_rollouts"])),
            "target_transform": transform,
            "lr_warmup_epochs": float(kwargs["lr_warmup_epochs"]),
            "raw_return_summary": {"count": 4.0},
            "target_summary": {"count": 4.0},
            "initial_value_return_correlation": -0.5,
            "final_value_return_correlation": correlation,
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollouts", fake_prefit_value_head_from_rollouts)

    artifact = phase3_macroplacement_ppo.run_ariane_critic_prefit_comparison(
        output_path=tmp_path / "phase3_ariane_critic_prefit_comparison.json",
        prefit_rollout_seeds=(0, 1),
    )

    assert len(prefit_calls) == 3
    assert [call["target_transform"] for call in prefit_calls] == ["raw", "raw", "zscore"]
    assert [call["epochs"] for call in prefit_calls] == [20, 200, 200]
    assert [call["lr_warmup_epochs"] for call in prefit_calls] == [0, 20, 20]
    assert artifact["conditions"]["raw_baseline"]["target_transform"] == "raw"
    assert artifact["conditions"]["raw_fair_shot"]["target_transform"] == "raw"
    assert artifact["conditions"]["zscore_fair_shot"]["target_transform"] == "zscore"
    assert artifact["prefit_gate"]["zscore_fair_shot"]["pass"] is True
    assert artifact["prefit_gate"]["raw_fair_shot"]["pass"] is False
    assert "return_to_go_vs_step_index" in artifact["simple_signal_diagnostic"]


def test_run_ariane_reverse_curriculum_probe_selects_low_hpwl_truncated_prefixes(monkeypatch, tmp_path: Path):
    curriculum_prefix_states = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    candidate_rollouts = {
        0: {"steps": 12, "final_hpwl": 2.90, "completed_episode": False, "termination_reason": "no_valid_actions", "transitions": [SimpleNamespace(action_index=0)] * 12},
        1: {"steps": 14, "final_hpwl": 3.10, "completed_episode": False, "termination_reason": "no_valid_actions", "transitions": [SimpleNamespace(action_index=1)] * 14},
        2: {"steps": 11, "final_hpwl": 2.80, "completed_episode": False, "termination_reason": "no_valid_actions", "transitions": [SimpleNamespace(action_index=2)] * 11},
        3: {"steps": 13, "final_hpwl": 2.70, "completed_episode": True, "termination_reason": "completed", "transitions": [SimpleNamespace(action_index=3)] * 13},
        4: {"steps": 5, "final_hpwl": 2.60, "completed_episode": False, "termination_reason": "no_valid_actions", "transitions": [SimpleNamespace(action_index=4)] * 5},
    }

    def fake_collect_policy_rollout(**kwargs):
        seed = kwargs["seed"]
        if seed in candidate_rollouts:
            rollout = dict(candidate_rollouts[seed])
            rollout.update({"invalid_action_count": 0, "episode_return": -rollout["final_hpwl"], "is_legal": True})
            return rollout
        return {
            "steps": 10,
            "final_hpwl": 2.75,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2.75,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=0)] * 10,
        }

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"prefix_seed_hpwl": rollout["final_hpwl"], "prefix_steps": prefix_steps}

    def fake_train_ppo_policy(**kwargs):
        initial_state_factory = kwargs.get("initial_state_factory")
        if initial_state_factory is not None:
            curriculum_prefix_states.append(initial_state_factory(0))
            curriculum_prefix_states.append(initial_state_factory(1))
        return {
            "history": [{"final_hpwl": 2.5, "best_hpwl_so_far": 2.5}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.5,
                "final_hpwl": 2.5,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)

    artifact = phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
        output_path=tmp_path / "phase3_reverse_curriculum_probe.json",
        base_updates=2,
        curriculum_updates=2,
        candidate_rollouts=5,
        top_k=2,
        tail_steps=5,
    )

    assert [row["seed"] for row in artifact["selected_prefix_pool"]] == [2, 0]
    assert [row["prefix_steps"] for row in artifact["selected_prefix_pool"]] == [6, 7]
    assert curriculum_prefix_states == [
        {"prefix_seed_hpwl": 2.80, "prefix_steps": 6},
        {"prefix_seed_hpwl": 2.90, "prefix_steps": 7},
    ]


def test_run_ariane_reverse_curriculum_probe_writes_status_artifact(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_collect_policy_rollout(**kwargs):
        return {
            "steps": 10,
            "final_hpwl": 2.75,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2.75,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=0)] * 10,
        }

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"prefix_seed_hpwl": rollout["final_hpwl"], "prefix_steps": prefix_steps}

    def fake_train_ppo_policy(**kwargs):
        return {
            "history": [{"final_hpwl": 2.5, "best_hpwl_so_far": 2.5}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.5,
                "final_hpwl": 2.5,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)

    output_path = tmp_path / "phase3_reverse_curriculum_probe.json"
    phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
        output_path=output_path,
        base_updates=2,
        curriculum_updates=2,
        candidate_rollouts=3,
        top_k=2,
        tail_steps=5,
    )

    status_path = tmp_path / "phase3_reverse_curriculum_probe_status.json"
    assert status_path.exists()
    status = __import__("json").loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["stage"] == "completed"
    assert status["selected_prefix_pool_size"] == 2


def test_run_ariane_reverse_curriculum_probe_records_failed_stage(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_train_ppo_policy(**kwargs):
        raise RuntimeError("synthetic_failure")

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)

    output_path = tmp_path / "phase3_reverse_curriculum_probe.json"
    try:
        phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
            output_path=output_path,
            base_updates=2,
            curriculum_updates=2,
            candidate_rollouts=3,
            top_k=2,
            tail_steps=5,
        )
    except RuntimeError as exc:
        assert str(exc) == "synthetic_failure"
    else:
        raise AssertionError("expected synthetic failure")

    status_path = tmp_path / "phase3_reverse_curriculum_probe_status.json"
    assert status_path.exists()
    status = __import__("json").loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stage"] == "base_training"
    assert "synthetic_failure" in status["error"]


def test_run_ariane_reverse_curriculum_probe_serializes_non_json_rollout_fields(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_collect_policy_rollout(**kwargs):
        return {
            "steps": 10,
            "final_hpwl": 2.75,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2.75,
            "is_legal": True,
            "schema": SimpleNamespace(name="feature_schema"),
            "transitions": [SimpleNamespace(action_index=0)],
        }

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"prefix_seed_hpwl": rollout["final_hpwl"], "prefix_steps": prefix_steps}

    def fake_train_ppo_policy(**kwargs):
        return {
            "history": [{"final_hpwl": 2.5, "best_hpwl_so_far": 2.5}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.5,
                "final_hpwl": 2.5,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
                "schema": SimpleNamespace(name="feature_schema"),
                "transitions": [SimpleNamespace(action_index=0)],
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)

    output_path = tmp_path / "phase3_reverse_curriculum_probe.json"
    artifact = phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
        output_path=output_path,
        base_updates=2,
        curriculum_updates=2,
        candidate_rollouts=3,
        top_k=2,
        tail_steps=5,
    )

    assert output_path.exists()
    payload = __import__("json").loads(output_path.read_text(encoding="utf-8"))
    assert payload["base_final_rollout"]["final_hpwl"] == 2.5
    assert "schema" not in payload["base_final_rollout"]
    assert "transitions" not in payload["base_final_rollout"]
    assert artifact["base_final_rollout"]["final_hpwl"] == 2.5


def test_run_ariane_reverse_curriculum_probe_v2_uses_external_prefix_source(monkeypatch, tmp_path: Path):
    policies = []
    train_calls = []
    curriculum_prefix_states = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_initialize_policy_for_problem(**kwargs):
        policy = object()
        policies.append(policy)
        return policy

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"name": rollout["name"], "prefix_steps": prefix_steps}

    def fake_evaluate_prefix_candidate_salvage(*, prefix_state, **kwargs):
        if prefix_state["name"] == "good_a":
            return {
                "greedy": {"completed_episode": True, "final_hpwl": 2.70},
                "random": {"completed_episode": False, "final_hpwl": 3.10},
            }
        if prefix_state["name"] == "good_b":
            return {
                "greedy": {"completed_episode": True, "final_hpwl": 2.82},
                "random": {"completed_episode": False, "final_hpwl": 3.00},
            }
        return {
            "greedy": {"completed_episode": False, "final_hpwl": 3.40},
            "random": {"completed_episode": False, "final_hpwl": 3.50},
        }

    def fake_train_ppo_policy(**kwargs):
        train_calls.append(kwargs)
        initial_state_factory = kwargs.get("initial_state_factory")
        if initial_state_factory is not None:
            curriculum_prefix_states.append(initial_state_factory(0))
            curriculum_prefix_states.append(initial_state_factory(1))
        return {
            "history": [{"final_hpwl": 2.5, "best_hpwl_so_far": 2.5}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.5,
                "final_hpwl": 2.5,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    prefix_source = tmp_path / "external_prefix_source.json"
    prefix_source.write_text(
        __import__("json").dumps(
            {
                "rollouts": [
                    {
                        "name": "too_weak",
                        "steps": 120,
                        "final_hpwl": 3.20,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [0] * 120,
                    },
                    {
                        "name": "good_b",
                        "steps": 123,
                        "final_hpwl": 2.84,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [1] * 123,
                    },
                    {
                        "name": "good_a",
                        "steps": 122,
                        "final_hpwl": 2.80,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [2] * 122,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "initialize_policy_for_problem", fake_initialize_policy_for_problem)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_evaluate_prefix_candidate_salvage", fake_evaluate_prefix_candidate_salvage)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)

    artifact = phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
        output_path=tmp_path / "phase3_reverse_curriculum_probe.json",
        curriculum_updates=2,
        candidate_rollouts=3,
        top_k=2,
        tail_steps=10,
        prefix_source_path=prefix_source,
        prefix_source_hpwl_threshold=2.85,
    )

    assert artifact["base_final_rollout"] is None
    assert artifact["prefix_source"]["mode"] == "external_artifact"
    assert [row["name"] for row in artifact["selected_prefix_pool"]] == ["good_a", "good_b"]
    assert all(row["salvage"]["greedy"]["completed_episode"] for row in artifact["selected_prefix_pool"])
    assert len(train_calls) == 2
    assert train_calls[0]["policy"] is not train_calls[1]["policy"]
    assert train_calls[0]["initial_state_factory"] is None
    assert curriculum_prefix_states == [
        {"name": "good_a", "prefix_steps": 112},
        {"name": "good_b", "prefix_steps": 113},
    ]


def test_run_ariane_reverse_curriculum_probe_v2_fails_without_salvageable_prefixes(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"name": rollout["name"], "prefix_steps": prefix_steps}

    def fake_evaluate_prefix_candidate_salvage(*, prefix_state, **kwargs):
        return {
            "greedy": {"completed_episode": False, "final_hpwl": 3.60},
            "random": {"completed_episode": False, "final_hpwl": 3.80},
        }

    prefix_source = tmp_path / "external_prefix_source.json"
    prefix_source.write_text(
        __import__("json").dumps(
            {
                "rollouts": [
                    {
                        "name": "bad_a",
                        "steps": 122,
                        "final_hpwl": 2.80,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [0] * 122,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_evaluate_prefix_candidate_salvage", fake_evaluate_prefix_candidate_salvage)

    output_path = tmp_path / "phase3_reverse_curriculum_probe.json"
    try:
        phase3_macroplacement_ppo.run_ariane_reverse_curriculum_probe(
            output_path=output_path,
            curriculum_updates=2,
            candidate_rollouts=1,
            top_k=1,
            tail_steps=10,
            prefix_source_path=prefix_source,
            prefix_source_hpwl_threshold=2.85,
        )
    except ValueError as exc:
        assert "salvageable" in str(exc)
    else:
        raise AssertionError("expected salvageable-prefix failure")


def test_export_ariane_prefix_pool_writes_replayable_action_indices(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 4,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: ("m0", "m1", "m2", "m3"))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_train_ppo_policy(**kwargs):
        return {
            "history": [{"final_hpwl": 2.9, "best_hpwl_so_far": 2.9}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.9,
                "final_hpwl": 2.9,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    def fake_collect_policy_rollout(**kwargs):
        seed = kwargs["seed"]
        return {
            "steps": 12 + seed,
            "final_hpwl": 2.8 + (0.1 * seed),
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -3.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=seed), SimpleNamespace(action_index=seed + 1)],
            "schema": SimpleNamespace(name="feature_schema"),
        }

    def fake_summarize_export_rollout_trace(**kwargs):
        rollout = kwargs["rollout"]
        return {
            "action_indices": [int(step.action_index) for step in rollout["transitions"]],
            "hpwl_trace": [11.0, 12.0],
            "reward_trace": [-1.0, -1.5],
            "transition_count": 2,
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "_summarize_export_rollout_trace",
        fake_summarize_export_rollout_trace,
        raising=False,
    )

    artifact = phase3_macroplacement_ppo.export_ariane_prefix_pool(
        output_path=tmp_path / "phase3_ariane_prefix_pool.json",
        num_updates=2,
        candidate_rollouts=2,
    )

    assert artifact["prefix_source"]["mode"] == "exported_policy_rollouts"
    assert artifact["base_final_rollout"]["final_hpwl"] == 2.9
    assert [row["action_indices"] for row in artifact["rollouts"]] == [[0, 1], [1, 2]]
    assert [row["seed"] for row in artifact["rollouts"]] == [0, 1]
    assert [row["rollout_trace"]["hpwl_trace"] for row in artifact["rollouts"]] == [[11.0, 12.0], [11.0, 12.0]]
    assert [row["rollout_trace"]["transition_count"] for row in artifact["rollouts"]] == [2, 2]


def test_export_ariane_prefix_pool_records_acceptance_summary(monkeypatch, tmp_path: Path):
    train_calls = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_train_ppo_policy(**kwargs):
        train_calls.append(kwargs["seed"])
        return {
            "history": [{"final_hpwl": 2.9, "best_hpwl_so_far": 2.9}],
            "final_rollout": {
                "steps": 12,
                "episode_return": -2.9,
                "final_hpwl": 2.9,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    rollout_rows = {
        (0, 0): {"steps": 122, "final_hpwl": 2.80, "termination_reason": "no_valid_actions"},
        (0, 1): {"steps": 121, "final_hpwl": 2.84, "termination_reason": "no_valid_actions"},
        (1, 0): {"steps": 118, "final_hpwl": 2.95, "termination_reason": "no_valid_actions"},
        (1, 1): {"steps": 124, "final_hpwl": 2.82, "termination_reason": "no_valid_actions"},
    }

    def fake_collect_policy_rollout(**kwargs):
        key = (kwargs["policy_seed"], kwargs["seed"])
        row = rollout_rows[key]
        return {
            "steps": row["steps"],
            "final_hpwl": row["final_hpwl"],
            "completed_episode": False,
            "termination_reason": row["termination_reason"],
            "invalid_action_count": 0,
            "episode_return": -row["final_hpwl"],
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=0), SimpleNamespace(action_index=1)],
            "schema": SimpleNamespace(name="feature_schema"),
        }

    def fake_summarize_rollout_geometry_from_rollout(*, rollout, **kwargs):
        remaining = 133 - int(rollout["steps"])
        ratio = 1.4 if float(rollout["final_hpwl"]) <= 2.85 else 0.8
        return {
            "remaining_macro_count": remaining,
            "free_space": {
                "largest_component_vs_remaining_area_ratio": ratio,
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_summarize_rollout_geometry_from_rollout", fake_summarize_rollout_geometry_from_rollout)

    artifact = phase3_macroplacement_ppo.export_ariane_prefix_pool(
        output_path=tmp_path / "phase3_ariane_prefix_pool.json",
        num_updates=2,
        candidate_rollouts=2,
        policy_seeds=(0, 1),
        hpwl_threshold=2.85,
        remaining_macro_range=(8, 13),
        hpwl_threshold_sweep=(2.85, 2.90),
        remaining_macro_ranges_sweep=((8, 13), (10, 15)),
        geometry_packing_factor=1.0,
        min_accepted_prefixes=3,
    )

    assert train_calls == [0, 1]
    assert artifact["prefix_source"]["mode"] == "exported_policy_rollouts"
    assert artifact["prefix_source"]["policy_seed_count"] == 2
    assert artifact["acceptance_criteria"]["hpwl_threshold"] == 2.85
    assert artifact["acceptance_summary"]["total_rollouts"] == 4
    assert artifact["acceptance_summary"]["accepted_prefixes"] == 3
    assert artifact["acceptance_summary"]["unique_source_rollouts"] == 3
    assert artifact["acceptance_summary"]["meets_min_prefix_count"] is True
    assert artifact["acceptance_summary"]["status"] == "pass"
    assert artifact["acceptance_summary"]["threshold_sweep"] == [
        {
            "accepted_prefixes": 3,
            "hpwl_threshold": 2.85,
            "remaining_macro_range": [8, 13],
            "unique_source_rollouts": 3,
        },
        {
            "accepted_prefixes": 2,
            "hpwl_threshold": 2.85,
            "remaining_macro_range": [10, 15],
            "unique_source_rollouts": 2,
        },
        {
            "accepted_prefixes": 3,
            "hpwl_threshold": 2.9,
            "remaining_macro_range": [8, 13],
            "unique_source_rollouts": 3,
        },
        {
            "accepted_prefixes": 2,
            "hpwl_threshold": 2.9,
            "remaining_macro_range": [10, 15],
            "unique_source_rollouts": 2,
        },
    ]
    accepted = [row for row in artifact["rollouts"] if row["accepted_for_reverse_curriculum"]]
    assert [(row["policy_seed"], row["seed"]) for row in accepted] == [(0, 0), (0, 1), (1, 1)]


def test_calibrate_ariane_prefix_pool_records_per_prefix_salvage(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"name": rollout["name"], "prefix_steps": prefix_steps}

    def fake_evaluate_prefix_candidate_salvage(*, prefix_state, **kwargs):
        if prefix_state["name"] == "good":
            return {
                "greedy": {"completed_episode": True, "final_hpwl": 2.70},
                "random": {"completed_episode": False, "final_hpwl": 3.10},
            }
        return {
            "greedy": {"completed_episode": False, "final_hpwl": 3.40},
            "random": {"completed_episode": False, "final_hpwl": 3.50},
        }

    prefix_source = tmp_path / "external_prefix_source.json"
    prefix_source.write_text(
        __import__("json").dumps(
            {
                "rollouts": [
                    {
                        "name": "good",
                        "steps": 122,
                        "final_hpwl": 2.80,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [0] * 122,
                    },
                    {
                        "name": "bad",
                        "steps": 123,
                        "final_hpwl": 2.83,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [1] * 123,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_evaluate_prefix_candidate_salvage", fake_evaluate_prefix_candidate_salvage)

    artifact = phase3_macroplacement_ppo.calibrate_ariane_prefix_pool(
        output_path=tmp_path / "phase3_ariane_prefix_calibration.json",
        prefix_source_path=prefix_source,
        tail_steps=10,
        hpwl_threshold=2.85,
    )

    assert artifact["prefix_source"]["mode"] == "external_artifact"
    assert artifact["summary"]["total_candidates"] == 2
    assert artifact["summary"]["greedy_completable"] == 1
    assert artifact["summary"]["random_completable"] == 0
    assert artifact["summary"]["recommended_salvage_mode"] == "greedy_or_random"
    assert [row["name"] for row in artifact["prefixes"] if row["salvage"]["greedy"]["completed_episode"]] == ["good"]


def test_hpwl_threshold_helpers_interpret_million_scale_and_raw_units_consistently():
    rollout = {
        "final_hpwl": 2_850_000.0,
        "completed_episode": False,
    }
    geometry_summary = {
        "remaining_macro_count": 12,
        "free_space": {"largest_component_vs_remaining_area_ratio": 1.25},
    }

    million_scale = phase3_macroplacement_ppo._evaluate_prefix_acceptance(
        rollout=rollout,
        geometry_summary=geometry_summary,
        hpwl_threshold=2.85,
        remaining_macro_range=(8, 13),
        geometry_packing_factor=1.0,
    )
    raw_scale = phase3_macroplacement_ppo._evaluate_prefix_acceptance(
        rollout=rollout,
        geometry_summary=geometry_summary,
        hpwl_threshold=2_850_000.0,
        remaining_macro_range=(8, 13),
        geometry_packing_factor=1.0,
    )
    tighter = phase3_macroplacement_ppo._evaluate_prefix_acceptance(
        rollout=rollout,
        geometry_summary=geometry_summary,
        hpwl_threshold=2.84,
        remaining_macro_range=(8, 13),
        geometry_packing_factor=1.0,
    )

    assert million_scale["hpwl_ok"] is True
    assert raw_scale["hpwl_ok"] is True
    assert million_scale["accepted_for_reverse_curriculum"] is True
    assert raw_scale["accepted_for_reverse_curriculum"] is True
    assert tighter["hpwl_ok"] is False

    hit_rates = phase3_macroplacement_ppo._build_hpwl_threshold_hit_rates(
        rollouts=[
            {"final_hpwl": 2_850_000.0},
            {"final_hpwl": 2_910_000.0},
        ],
        hpwl_threshold_sweep=(2.85, 2_850_000.0, 2.92),
    )

    assert hit_rates == [
        {"count": 1, "hpwl_threshold": 2.85, "hit_rate": 0.5},
        {"count": 1, "hpwl_threshold": 2_850_000.0, "hit_rate": 0.5},
        {"count": 2, "hpwl_threshold": 2.92, "hit_rate": 1.0},
    ]


def test_calibrate_ariane_prefix_pool_interprets_million_scale_thresholds(monkeypatch, tmp_path: Path):
    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"name": rollout["name"], "prefix_steps": prefix_steps}

    def fake_evaluate_prefix_candidate_salvage(*, prefix_state, **kwargs):
        return {
            "greedy": {"completed_episode": prefix_state["name"] != "too_high", "final_hpwl": 2_700_000.0},
            "random": {"completed_episode": False, "final_hpwl": 3_100_000.0},
        }

    prefix_source = tmp_path / "external_prefix_source_realistic.json"
    prefix_source.write_text(
        __import__("json").dumps(
            {
                "rollouts": [
                    {
                        "name": "good_low",
                        "steps": 122,
                        "final_hpwl": 2_800_000.0,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [0] * 122,
                    },
                    {
                        "name": "good_edge",
                        "steps": 123,
                        "final_hpwl": 2_850_000.0,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [1] * 123,
                    },
                    {
                        "name": "too_high",
                        "steps": 124,
                        "final_hpwl": 2_930_000.0,
                        "completed_episode": False,
                        "termination_reason": "no_valid_actions",
                        "action_indices": [2] * 124,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_evaluate_prefix_candidate_salvage", fake_evaluate_prefix_candidate_salvage)

    artifact = phase3_macroplacement_ppo.calibrate_ariane_prefix_pool(
        output_path=tmp_path / "phase3_ariane_prefix_calibration_realistic.json",
        prefix_source_path=prefix_source,
        tail_steps=10,
        hpwl_threshold=2.85,
    )

    assert artifact["prefix_source"]["eligible_rollout_count"] == 2
    assert artifact["summary"]["total_candidates"] == 2
    assert [row["name"] for row in artifact["prefixes"]] == ["good_low", "good_edge"]
    assert artifact["summary"]["greedy_completable"] == 2


def test_export_ariane_random_prefix_pool_records_distribution_and_full_rollouts(monkeypatch, tmp_path: Path):
    collect_calls = []

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 3,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), object(), reward_model, adapter, metadata

    rollout_map = {
        0: {
            "steps": 121,
            "final_hpwl": 2_840_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2_840_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=0)],
            "schema": None,
        },
        1: {
            "steps": 117,
            "final_hpwl": 2_910_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2_910_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=1)],
            "schema": None,
        },
        2: {
            "steps": 124,
            "final_hpwl": 3_050_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -3_050_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=2)],
            "schema": None,
        },
    }

    def fake_collect_policy_rollout(**kwargs):
        collect_calls.append(kwargs)
        return dict(rollout_map[kwargs["seed"]])

    geometry_map = {
        0: {"remaining_macro_count": 12, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
        1: {"remaining_macro_count": 16, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
        2: {"remaining_macro_count": 9, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
    }

    def fake_summarize_geometry(*, rollout, **kwargs):
        for seed, row in rollout_map.items():
            if row["final_hpwl"] == rollout["final_hpwl"]:
                return geometry_map[seed]
        raise AssertionError("unexpected rollout")

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_summarize_rollout_geometry_from_rollout", fake_summarize_geometry)
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "_summarize_export_rollout_trace",
        lambda **kwargs: {"action_indices": [1, 2], "hpwl_trace": [11.0, 12.0], "reward_trace": [0.5], "value_trace": [0.25], "transition_count": 1},
        raising=False,
    )

    artifact = phase3_macroplacement_ppo.export_ariane_random_prefix_pool(
        output_path=tmp_path / "phase3_ariane_random_prefix_pool.json",
        num_rollouts=3,
        hpwl_threshold=2.85,
        remaining_macro_range=(8, 13),
        hpwl_threshold_sweep=(2.85, 2.92),
        remaining_macro_ranges_sweep=((8, 13), (15, 20)),
        geometry_packing_factor=1.0,
        min_accepted_prefixes=2,
    )

    assert [call["uniform_legal_policy"] for call in collect_calls] == [True, True, True]
    assert artifact["prefix_source"]["mode"] == "random_uniform_legal_rollouts"
    assert artifact["acceptance_summary"]["accepted_prefixes"] == 1
    assert artifact["acceptance_summary"]["unique_source_rollouts"] == 1
    assert artifact["acceptance_summary"]["status"] == "partial"
    assert artifact["acceptance_summary"]["threshold_sweep"] == [
        {"accepted_prefixes": 1, "hpwl_threshold": 2.85, "remaining_macro_range": [8, 13], "unique_source_rollouts": 1},
        {"accepted_prefixes": 0, "hpwl_threshold": 2.85, "remaining_macro_range": [15, 20], "unique_source_rollouts": 0},
        {"accepted_prefixes": 1, "hpwl_threshold": 2.92, "remaining_macro_range": [8, 13], "unique_source_rollouts": 1},
        {"accepted_prefixes": 1, "hpwl_threshold": 2.92, "remaining_macro_range": [15, 20], "unique_source_rollouts": 1},
    ]
    assert artifact["distribution"]["final_hpwl"]["min"] == 2_840_000.0
    assert artifact["distribution"]["final_hpwl"]["max"] == 3_050_000.0
    assert artifact["distribution"]["threshold_hit_rates"] == [
        {"count": 1, "hpwl_threshold": 2.85, "hit_rate": 1 / 3},
        {"count": 2, "hpwl_threshold": 2.92, "hit_rate": 2 / 3},
    ]
    assert artifact["rollouts"][0]["rollout_trace"]["action_indices"] == [1, 2]


def test_export_ariane_warm_start_probe_records_update0_update1_and_salvage(monkeypatch, tmp_path: Path):
    collect_calls = []
    train_calls = []
    warm_start_calls = []

    class _Policy:
        def __init__(self, label):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), _Policy("base"), reward_model, adapter, metadata

    update0_rollouts = {
        0: {
            "steps": 121,
            "final_hpwl": 2_810_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2_810_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=0)] * 121,
            "schema": None,
        },
        1: {
            "steps": 128,
            "final_hpwl": 2_900_000.0,
            "completed_episode": True,
            "termination_reason": "completed",
            "invalid_action_count": 0,
            "episode_return": -2_900_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=1)] * 128,
            "schema": None,
        },
    }
    update1_rollouts = {
        100: {
            "steps": 119,
            "final_hpwl": 3_050_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -3_050_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=2)] * 119,
            "schema": None,
        },
        101: {
            "steps": 120,
            "final_hpwl": 3_100_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -3_100_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(action_index=3)] * 120,
            "schema": None,
        },
    }

    def fake_collect_policy_rollout(**kwargs):
        collect_calls.append(kwargs)
        seed = kwargs["seed"]
        policy = kwargs["policy"]
        if policy.label == "warm0":
            return dict(update0_rollouts[seed])
        return dict(update1_rollouts[seed])

    def fake_initialize_policy_for_problem(**kwargs):
        del kwargs
        return _Policy("init")

    def fake_load_policy_warm_start(policy, checkpoint):
        warm_start_calls.append((policy.label, checkpoint))
        policy.label = "warm0" if len(warm_start_calls) == 1 else "warm1"
        return True

    def fake_train_ppo_policy(**kwargs):
        train_calls.append(kwargs)
        kwargs["policy"].label = "trained1"
        return {
            "history": [{"final_hpwl": 3.2, "best_hpwl_so_far": 3.2}],
            "final_rollout": {
                "steps": 120,
                "episode_return": -3.2,
                "final_hpwl": 3.2,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    geometry_by_hpwl = {
        2_810_000.0: {"remaining_macro_count": 12, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
        2_900_000.0: {"remaining_macro_count": 9, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
        3_050_000.0: {"remaining_macro_count": 13, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
        3_100_000.0: {"remaining_macro_count": 14, "free_space": {"largest_component_vs_remaining_area_ratio": 1.5}},
    }

    def fake_summarize_geometry(*, rollout, **kwargs):
        return geometry_by_hpwl[float(rollout["final_hpwl"])]

    def fake_replay_prefix_state_from_rollout(*, rollout, prefix_steps, **kwargs):
        return {"name": rollout["source_rollout_id"], "prefix_steps": prefix_steps}

    def fake_evaluate_prefix_candidate_salvage(*, prefix_state, **kwargs):
        if prefix_state["name"] == "update0:0":
            return {
                "greedy": {"completed_episode": True, "final_hpwl": 2.7},
                "random": {"completed_episode": False, "final_hpwl": 3.0},
            }
        return {
            "greedy": {"completed_episode": False, "final_hpwl": 3.3},
            "random": {"completed_episode": False, "final_hpwl": 3.4},
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "initialize_policy_for_problem", fake_initialize_policy_for_problem)
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", fake_load_policy_warm_start)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_summarize_rollout_geometry_from_rollout", fake_summarize_geometry)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_replay_prefix_state_from_rollout", fake_replay_prefix_state_from_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_evaluate_prefix_candidate_salvage", fake_evaluate_prefix_candidate_salvage)
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "_summarize_export_rollout_trace",
        lambda **kwargs: {"action_indices": [1, 2], "hpwl_trace": [11.0, 12.0], "reward_trace": [0.5], "value_trace": [0.25], "transition_count": 1},
        raising=False,
    )
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")

    artifact = phase3_macroplacement_ppo.export_ariane_warm_start_probe(
        output_path=tmp_path / "phase3_ariane_warm_start_probe.json",
        update0_rollouts=2,
        update1_rollouts=2,
        hpwl_threshold=2.85,
        remaining_macro_range=(8, 13),
        hpwl_threshold_sweep=(2.85, 2.92),
        remaining_macro_ranges_sweep=((8, 13), (10, 15)),
        tail_steps=10,
    )

    assert len(train_calls) == 1
    assert [call["uniform_legal_policy"] for call in collect_calls] == [False, False, False, False]
    assert artifact["cohorts"]["update0"]["distribution"]["final_hpwl"]["min"] == 2_810_000.0
    assert artifact["cohorts"]["update0"]["completion_rate"] == 0.5
    assert artifact["cohorts"]["update1"]["distribution"]["final_hpwl"]["min"] == 3_050_000.0
    assert artifact["cohorts"]["update1"]["completion_rate"] == 0.0
    assert artifact["cohorts"]["update0"]["acceptance_summary"]["accepted_prefixes"] == 1
    assert artifact["cohorts"]["update1"]["acceptance_summary"]["accepted_prefixes"] == 0
    assert artifact["salvage_calibration"]["summary"]["total_candidates"] == 1
    assert artifact["salvage_calibration"]["summary"]["greedy_completable"] == 1
    assert artifact["salvage_calibration"]["prefixes"][0]["source_rollout_id"] == "update0:0"
    assert train_calls[0]["reference_policy"] is not None
    assert train_calls[0]["kl_anchor_beta"] == 5.0


def test_run_ariane_warm_start_substep_diagnostic_records_substep_metrics(monkeypatch, tmp_path: Path):
    run_single_update_calls = []
    warm_start_calls = []

    class _Policy:
        def __init__(self, label):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), _Policy("base"), reward_model, adapter, metadata

    def fake_load_policy_warm_start(policy, checkpoint):
        warm_start_calls.append((policy.label, checkpoint))
        policy.label = "warm"
        return True

    def fake_run_single_ppo_update(**kwargs):
        run_single_update_calls.append(kwargs)
        return {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "value_loss_train_space": 0.2,
            "value_loss_raw_space": 10.0,
            "entropy": 0.3,
            "steps": 120.0,
            "final_hpwl": 2_950_000.0,
            "invalid_action_count": 0.0,
            "episode_return": -2_950_000.0,
            "completed_episode": 0.0,
            "is_legal": 1.0,
            "approx_kl": 0.01,
            "kl_anchor_loss": 0.0,
            "kl_anchor_beta": 5.0,
            "pre_update_diagnostic": {
                "heldout_rollout_count": 3,
                "heldout_completion_rate": 1 / 3,
                "heldout_mean_hpwl": 2_880_000.0,
                "heldout_log_prob_delta_mean": 0.0,
                "update_batch_kl_to_reference": 0.0,
            },
            "pre_update_metrics": {
                "entropy": 0.5,
            },
            "batch_diagnostic": {
                "return_summary": {"count": 120, "mean": -10.0},
                "old_value_summary": {"count": 120, "mean": -2.0},
                "raw_advantage_summary": {"count": 120, "mean": -8.0},
                "normalized_advantage_summary": {"count": 120, "mean": 0.0},
                "top_abs_advantages": [{"transition_index": 4, "raw_advantage": 12.0}],
                "value_return_correlation": 0.1,
                "advantage_gate": {
                    "std_gt_epsilon": True,
                    "mean_over_std_within_band": True,
                    "pass": True,
                },
            },
            "value_target_stats": {"mean": 0.0, "std": 1.0},
            "substep_history": [
                {
                    "substep_index": 0,
                    "heldout_rollout_count": 3,
                    "heldout_completion_rate": 0.0,
                    "heldout_mean_hpwl": 2_990_000.0,
                    "heldout_log_prob_delta_mean": -0.2,
                    "update_batch_kl_to_reference": 0.15,
                    "global_grad_norm": 3.0,
                    "policy_head_grad_norm": 2.0,
                    "value_head_grad_norm": 1.0,
                    "value_loss_train_space": 0.2,
                    "value_loss_raw_space": 10.0,
                    "value_target_mean": 0.0,
                    "value_target_std": 1.0,
                    "grad_clip_norm": None,
                    "global_grad_norm_pre_clip": 3.0,
                    "global_grad_norm_post_clip": 3.0,
                    "grad_clip_applied": False,
                    "effective_learning_rate": 1e-4,
                    "first_substep_lr_scale": 0.1,
                    "global_param_delta_norm_from_pre_update": 0.02,
                    "policy_head_param_delta_norm_from_pre_update": 0.01,
                    "ratio_fraction_gt_1_5": 0.2,
                    "ratio_fraction_lt_0_67": 0.1,
                    "ratio_max": 1.8,
                    "approx_kl_abs_top10pct_fraction": 0.95,
                    "entropy_delta_from_pre_update": -0.2,
                },
                {
                    "substep_index": 1,
                    "heldout_rollout_count": 3,
                    "heldout_completion_rate": 0.0,
                    "heldout_mean_hpwl": 3_010_000.0,
                    "heldout_log_prob_delta_mean": -0.4,
                    "update_batch_kl_to_reference": 0.22,
                    "global_grad_norm": 2.5,
                    "policy_head_grad_norm": 1.5,
                    "value_head_grad_norm": 0.9,
                    "value_loss_train_space": 0.25,
                    "value_loss_raw_space": 11.0,
                    "value_target_mean": 0.0,
                    "value_target_std": 1.0,
                    "grad_clip_norm": None,
                    "global_grad_norm_pre_clip": 2.5,
                    "global_grad_norm_post_clip": 2.5,
                    "grad_clip_applied": False,
                    "effective_learning_rate": 1e-3,
                    "first_substep_lr_scale": 0.1,
                    "global_param_delta_norm_from_pre_update": 0.04,
                    "policy_head_param_delta_norm_from_pre_update": 0.02,
                    "ratio_fraction_gt_1_5": 0.25,
                    "ratio_fraction_lt_0_67": 0.12,
                    "ratio_max": 1.9,
                    "approx_kl_abs_top10pct_fraction": 0.96,
                    "entropy_delta_from_pre_update": -0.3,
                },
            ],
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", fake_load_policy_warm_start)
    monkeypatch.setattr(phase3_macroplacement_ppo, "run_single_ppo_update", fake_run_single_ppo_update)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")

    artifact = phase3_macroplacement_ppo.run_ariane_warm_start_substep_diagnostic(
        output_path=tmp_path / "phase3_ariane_warm_start_substep_diagnostic.json",
        policy_epochs=2,
        diagnostic_eval_seeds=(7, 8, 9),
        first_substep_lr_scale=0.1,
    )

    assert len(run_single_update_calls) == 1
    assert run_single_update_calls[0]["policy_epochs"] == 2
    assert run_single_update_calls[0]["diagnostic_eval_seeds"] == (7, 8, 9)
    assert run_single_update_calls[0]["first_substep_lr_scale"] == 0.1
    assert run_single_update_calls[0]["reference_policy"] is not None
    assert run_single_update_calls[0]["kl_anchor_beta"] == 5.0
    assert artifact["diagnostic_config"]["policy_epochs"] == 2
    assert artifact["diagnostic_config"]["diagnostic_eval_seeds"] == [7, 8, 9]
    assert artifact["update_metrics"]["pre_update_diagnostic"]["heldout_completion_rate"] == 1 / 3
    assert artifact["update_metrics"]["batch_diagnostic"]["return_summary"]["count"] == 120
    assert artifact["update_metrics"]["batch_diagnostic"]["advantage_gate"]["pass"] is True
    assert artifact["update_metrics"]["value_target_stats"] == {"mean": 0.0, "std": 1.0}
    assert artifact["update_metrics"]["batch_diagnostic"]["top_abs_advantages"][0]["transition_index"] == 4
    assert artifact["update_metrics"]["substep_history"][0]["substep_index"] == 0
    assert artifact["update_metrics"]["substep_history"][1]["update_batch_kl_to_reference"] == 0.22
    assert artifact["update_metrics"]["substep_history"][0]["global_grad_norm"] == 3.0
    assert artifact["update_metrics"]["substep_history"][0]["value_loss_raw_space"] == 10.0
    assert artifact["update_metrics"]["substep_history"][0]["grad_clip_applied"] is False
    assert artifact["update_metrics"]["substep_history"][0]["ratio_fraction_gt_1_5"] == 0.2
    assert artifact["update_metrics"]["substep_history"][0]["approx_kl_abs_top10pct_fraction"] == 0.95
    assert len(warm_start_calls) == 1


def test_run_ariane_critic_prefit_diagnostic_records_prefit_and_gate_status(monkeypatch, tmp_path: Path):
    collect_calls = []
    prefit_calls = []
    update_calls = []
    train_calls = []
    warm_start_calls = []

    class _Policy:
        def __init__(self, label):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), _Policy("base"), reward_model, adapter, metadata

    def fake_load_policy_warm_start(policy, checkpoint):
        warm_start_calls.append((policy.label, checkpoint))
        policy.label = "warm"
        return True

    def fake_collect_policy_rollout(**kwargs):
        collect_calls.append(kwargs)
        seed = int(kwargs["seed"])
        return {
            "steps": 120,
            "final_hpwl": 2_880_000.0 + seed,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2_880_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(observation={"dummy": 1}, action_index=0, log_prob=-0.1, reward=1.0, value=0.0)] * 3,
            "schema": "schema",
        }

    def fake_prefit_value_head_from_rollouts(**kwargs):
        prefit_calls.append(kwargs)
        return {
            "initial_loss": 100.0,
            "final_loss": 10.0,
            "epochs": 20.0,
            "rollout_count": 4.0,
            "transition_count": 12.0,
            "target_transform": kwargs["target_transform"],
        }

    def fake_run_single_ppo_update(**kwargs):
        update_calls.append(kwargs)
        return {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "value_loss_train_space": 0.2,
            "value_loss_raw_space": 12.0,
            "entropy": 0.3,
            "steps": 120.0,
            "final_hpwl": 2_950_000.0,
            "invalid_action_count": 0.0,
            "episode_return": -2_950_000.0,
            "completed_episode": 0.0,
            "is_legal": 1.0,
            "approx_kl": 0.01,
            "kl_anchor_loss": 0.0,
            "kl_anchor_beta": 5.0,
            "pre_update_diagnostic": {
                "heldout_rollout_count": 5,
                "heldout_completion_rate": 0.0,
                "heldout_mean_hpwl": 2_870_000.0,
                "heldout_log_prob_delta_mean": 0.0,
                "update_batch_kl_to_reference": 0.0,
            },
            "batch_diagnostic": {
                "return_summary": {"count": 12, "mean": -100.0},
                "old_value_summary": {"count": 12, "mean": -10.0},
                "raw_advantage_summary": {"count": 12, "mean": -90.0},
                "normalized_advantage_summary": {"count": 12, "mean": 0.0},
                "top_abs_advantages": [{"transition_index": 1, "raw_advantage": 9.0}],
                "value_return_correlation": 0.5,
                "advantage_gate": {
                    "std_gt_epsilon": True,
                    "mean_over_std_within_band": True,
                    "pass": True,
                },
            },
            "value_target_stats": {"mean": 10.0, "std": 2.0},
            "substep_history": [
                {
                    "substep_index": 0,
                    "heldout_rollout_count": 5,
                    "heldout_completion_rate": 0.0,
                    "heldout_mean_hpwl": 2_885_000.0,
                    "heldout_log_prob_delta_mean": -0.04,
                    "update_batch_kl_to_reference": 0.5,
                    "global_grad_norm": 3.0,
                    "policy_head_grad_norm": 2.0,
                    "value_head_grad_norm": 1.0,
                    "value_loss_train_space": 0.2,
                    "value_loss_raw_space": 12.0,
                    "value_target_mean": 10.0,
                    "value_target_std": 2.0,
                    "grad_clip_norm": 30.0,
                    "global_grad_norm_pre_clip": 40.0,
                    "global_grad_norm_post_clip": 30.0,
                    "grad_clip_applied": True,
                    "effective_learning_rate": 1e-4,
                    "first_substep_lr_scale": 0.1,
                    "global_param_delta_norm_from_pre_update": 0.02,
                    "policy_head_param_delta_norm_from_pre_update": 0.01,
                    "ratio_fraction_gt_1_5": 0.2,
                    "ratio_fraction_lt_0_67": 0.1,
                    "ratio_max": 1.8,
                    "approx_kl_abs_top10pct_fraction": 0.95,
                    "entropy_delta_from_pre_update": -0.1,
                }
            ],
        }

    def fake_train_ppo_policy(**kwargs):
        train_calls.append(kwargs)
        return {
            "history": [
                {"final_hpwl": 2_900_000.0, "best_hpwl_so_far": 2_900_000.0},
                {"final_hpwl": 2_930_000.0, "best_hpwl_so_far": 2_900_000.0},
            ],
            "final_rollout": {
                "steps": 120,
                "episode_return": -2_930_000.0,
                "final_hpwl": 2_930_000.0,
                "invalid_action_count": 0,
                "termination_reason": "no_valid_actions",
                "completed_episode": False,
                "is_legal": True,
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", fake_load_policy_warm_start)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollouts", fake_prefit_value_head_from_rollouts, raising=False)
    monkeypatch.setattr(phase3_macroplacement_ppo, "run_single_ppo_update", fake_run_single_ppo_update)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")

    artifact = phase3_macroplacement_ppo.run_ariane_critic_prefit_diagnostic(
        output_path=tmp_path / "phase3_ariane_critic_prefit_diagnostic.json",
        prefit_rollout_seeds=(0, 1, 2, 3),
        diagnostic_eval_seeds=(10, 11, 12, 13, 14),
        prefit_epochs=20,
        prefit_lr_warmup_epochs=5,
        policy_epochs=1,
        followup_updates=10,
        warm_start_baseline_hpwl=2_881_800.0,
        target_transform="zscore",
        grad_clip_norm=30.0,
        first_substep_lr_scale=0.1,
    )

    assert len(warm_start_calls) == 1
    assert [call["seed"] for call in collect_calls[:4]] == [0, 1, 2, 3]
    assert prefit_calls[0]["epochs"] == 20
    assert prefit_calls[0]["lr_warmup_epochs"] == 5
    assert prefit_calls[0]["target_transform"] == "zscore"
    assert prefit_calls[0]["fixed_rollouts"][0].final_hpwl == 2_880_000.0
    assert update_calls[0]["policy_epochs"] == 1
    assert update_calls[0]["grad_clip_norm"] == 30.0
    assert update_calls[0]["first_substep_lr_scale"] == 0.1
    assert update_calls[0]["diagnostic_eval_seeds"] == (10, 11, 12, 13, 14)
    assert train_calls[0]["num_updates"] == 10
    assert artifact["prefit_metrics"]["final_loss"] == 10.0
    assert artifact["metadata"]["training_config"]["grad_clip_norm"] == 30.0
    assert artifact["metadata"]["warm_value_prefit_config"]["target_transform"] == "zscore"
    assert artifact["metadata"]["warm_value_prefit_config"]["prefit_lr_warmup_epochs"] == 5
    assert artifact["gate_status"]["necessary"]["finite_kl"] is True
    assert artifact["gate_status"]["necessary"]["heldout_log_prob_delta_within_band"] is True
    assert artifact["gate_status"]["necessary"]["advantage_non_degenerate"] is True
    assert artifact["gate_status"]["severity"]["approx_kl_band"] == "clear"
    assert artifact["gate_status"]["severity"]["hpwl_drift_band"] == "partial"
    assert artifact["gate_status"]["sufficient"]["followup_hpwl_within_3pct_of_baseline"] is True


def test_run_ariane_critic_prefit_diagnostic_skips_followup_when_necessary_gate_fails(monkeypatch, tmp_path: Path):
    class _Policy:
        def __init__(self, label):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 4,
            "ordered_macro_count": 133,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        adapter = SimpleNamespace(ordered_macros=lambda: tuple(f"m{i}" for i in range(133)))
        return SimpleNamespace(name="problem"), _Policy("base"), reward_model, adapter, metadata

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda policy, checkpoint: True)
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "collect_policy_rollout",
        lambda **kwargs: {
            "steps": 10,
            "final_hpwl": 2_900_000.0,
            "completed_episode": False,
            "termination_reason": "no_valid_actions",
            "invalid_action_count": 0,
            "episode_return": -2_900_000.0,
            "is_legal": True,
            "transitions": [SimpleNamespace(observation={"dummy": 1}, action_index=0, log_prob=-0.1, reward=1.0, value=0.0)],
            "schema": "schema",
        },
    )
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "prefit_value_head_from_rollouts",
        lambda **kwargs: {"initial_loss": 1.0, "final_loss": 0.5, "epochs": 20.0, "rollout_count": 1.0, "transition_count": 1.0},
        raising=False,
    )
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "run_single_ppo_update",
        lambda **kwargs: {
            "policy_loss": 0.1,
            "value_loss": 0.2,
            "value_loss_train_space": 0.2,
            "value_loss_raw_space": 15.0,
            "entropy": 0.3,
            "steps": 10.0,
            "final_hpwl": 2_950_000.0,
            "invalid_action_count": 0.0,
            "episode_return": -2_950_000.0,
            "completed_episode": 0.0,
            "is_legal": 1.0,
            "approx_kl": 0.01,
            "kl_anchor_loss": 0.0,
            "kl_anchor_beta": 5.0,
            "pre_update_diagnostic": {
                "heldout_rollout_count": 5,
                "heldout_completion_rate": 0.0,
                "heldout_mean_hpwl": 2_870_000.0,
                "heldout_log_prob_delta_mean": 0.0,
                "update_batch_kl_to_reference": 0.0,
            },
            "batch_diagnostic": {
                "return_summary": {"count": 1, "mean": -1.0},
                "old_value_summary": {"count": 1, "mean": -1.0},
                "raw_advantage_summary": {"count": 1, "mean": -1.0},
                "normalized_advantage_summary": {"count": 1, "mean": 0.0},
                "top_abs_advantages": [{"transition_index": 0, "raw_advantage": 1.0}],
                "value_return_correlation": 0.0,
                "advantage_gate": {
                    "std_gt_epsilon": False,
                    "mean_over_std_within_band": False,
                    "pass": False,
                },
            },
            "value_target_stats": {"mean": 0.0, "std": 1.0},
            "substep_history": [
                {
                    "substep_index": 0,
                    "heldout_rollout_count": 5,
                    "heldout_completion_rate": 0.0,
                    "heldout_mean_hpwl": 2_990_000.0,
                    "heldout_log_prob_delta_mean": -0.2,
                    "update_batch_kl_to_reference": float("inf"),
                    "global_grad_norm": 3.0,
                    "policy_head_grad_norm": 2.0,
                    "value_head_grad_norm": 0.0,
                    "value_loss_train_space": 0.2,
                    "value_loss_raw_space": 15.0,
                    "value_target_mean": 0.0,
                    "value_target_std": 1.0,
                    "grad_clip_norm": None,
                    "global_grad_norm_pre_clip": 3.0,
                    "global_grad_norm_post_clip": 3.0,
                    "grad_clip_applied": False,
                    "effective_learning_rate": 1e-3,
                    "first_substep_lr_scale": None,
                    "global_param_delta_norm_from_pre_update": 0.04,
                    "policy_head_param_delta_norm_from_pre_update": 0.02,
                }
            ],
        },
    )
    train_called = {"value": False}

    def fake_train_ppo_policy(**kwargs):
        train_called["value"] = True
        return {}

    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")

    artifact = phase3_macroplacement_ppo.run_ariane_critic_prefit_diagnostic(
        output_path=tmp_path / "phase3_ariane_critic_prefit_diagnostic.json",
        prefit_rollout_seeds=(0,),
        diagnostic_eval_seeds=(10, 11),
        prefit_epochs=20,
        prefit_lr_warmup_epochs=0,
        policy_epochs=1,
        followup_updates=10,
        warm_start_baseline_hpwl=2_881_800.0,
    )

    assert artifact["gate_status"]["necessary"]["pass"] is False
    assert artifact["gate_status"]["necessary"]["advantage_non_degenerate"] is False
    assert artifact["followup_training"] is None
    assert artifact["followup_eval"] is None
    assert train_called["value"] is False


def test_run_ariane_argmax_agreement_diagnostic_records_baseline_global_agreement(monkeypatch, tmp_path: Path):
    collect_calls = []
    prefit_calls = []
    update_calls = []
    agreement_calls = []

    class _DummyPolicy:
        def __init__(self, label: str):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 2,
            "ordered_macro_count": 2,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0, "legality_penalty": 10.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), _DummyPolicy("warm_start"), reward_model, _StubAdapter(), metadata

    def fake_collect_policy_rollout(**kwargs):
        policy = kwargs["policy"]
        collect_calls.append(
            {
                "label": policy.label,
                "seed": kwargs["seed"],
                "deterministic": kwargs["deterministic"],
            }
        )
        return {
            "steps": 2,
            "invalid_action_count": 0,
            "transitions": [],
            "schema": "schema",
            "final_hpwl": 123.0 if kwargs["deterministic"] else 150.0,
            "episode_return": -123.0 if kwargs["deterministic"] else -150.0,
            "termination_reason": "completed",
            "completed_episode": True,
            "is_legal": True,
        }

    def fake_prefit_value_head_from_rollouts(**kwargs):
        prefit_calls.append(kwargs)
        return {
            "initial_loss": 4.0,
            "final_loss": 1.0,
            "epochs": float(kwargs["epochs"]),
            "target_transform": kwargs["target_transform"],
            "final_value_return_correlation": 0.85,
        }

    def fake_run_single_ppo_update(**kwargs):
        update_calls.append(kwargs)
        kwargs["policy"].label = "post_update"
        return {
            "episode_return": -100.0,
            "final_hpwl": 100.0,
            "substep_history": [
                {
                    "substep_index": 0.0,
                    "approx_kl": 0.4,
                    "update_batch_kl_to_reference": 0.4,
                }
            ],
        }

    def fake_summarize_argmax_agreement_for_fixed_rollout(**kwargs):
        agreement_calls.append(
            {
                "reference_label": kwargs["reference_policy"].label,
                "candidate_label": kwargs["candidate_policy"].label,
                "step_count": len(kwargs["fixed_rollout"].transitions),
            }
        )
        return {
            "step_count": 2,
            "agreement_count": 1,
            "flip_count": 1,
            "overall_agreement_rate": 0.5,
            "flipped_step_indices": [1],
            "position_buckets": {
                "early": {"step_count": 1, "agreement_count": 1, "flip_count": 0, "agreement_rate": 1.0},
                "mid": {"step_count": 0, "agreement_count": 0, "flip_count": 0, "agreement_rate": 0.0},
                "late": {"step_count": 1, "agreement_count": 0, "flip_count": 1, "agreement_rate": 0.0},
            },
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollouts", fake_prefit_value_head_from_rollouts)
    monkeypatch.setattr(phase3_macroplacement_ppo, "run_single_ppo_update", fake_run_single_ppo_update)
    monkeypatch.setattr(
        phase3_macroplacement_ppo,
        "summarize_argmax_agreement_for_fixed_rollout",
        fake_summarize_argmax_agreement_for_fixed_rollout,
    )

    artifact = phase3_macroplacement_ppo.run_ariane_argmax_agreement_diagnostic(
        output_path=tmp_path / "phase3_ariane_argmax_agreement_diagnostic.json",
        prefit_rollout_seeds=(3, 4),
        trace_seed=11,
    )

    assert collect_calls[0] == {"label": "warm_start", "seed": 11, "deterministic": True}
    assert collect_calls[1:] == [
        {"label": "warm_start", "seed": 3, "deterministic": False},
        {"label": "warm_start", "seed": 4, "deterministic": False},
    ]
    assert prefit_calls[0]["target_transform"] == "zscore"
    assert prefit_calls[0]["epochs"] == 200
    assert update_calls[0]["reference_policy"].label == "warm_start"
    assert agreement_calls[0]["reference_label"] == "warm_start"
    assert agreement_calls[0]["candidate_label"] == "post_update"
    assert artifact["reference_trace"]["seed"] == 11
    assert artifact["prefit_metrics"]["final_value_return_correlation"] == 0.85
    assert artifact["substep0_update"]["approx_kl"] == 0.4
    assert artifact["agreement_summary"]["overall_agreement_rate"] == 0.5
    assert artifact["diagnostic_config"]["prefit_rollout_seeds"] == [3, 4]


def test_select_fragile_decision_priority_subset_keeps_overlap_and_low_margin_unique():
    summary = phase3_macroplacement_ppo._select_fragile_decision_priority_subset(
        baseline_flipped_step_indices=[1, 2, 3, 4, 5, 6],
        alpha10_flipped_step_indices=[1, 3],
        pre_update_margin_by_step={
            1: 0.01,
            2: 0.04,
            3: 0.02,
            4: 0.10,
            5: 0.03,
            6: 0.20,
        },
        unique_budget=2,
    )

    assert summary["overlap_steps"] == [1, 3]
    assert summary["unique_steps"] == [5, 2]
    assert summary["evaluated_steps"] == [1, 3, 5, 2]


def test_summarize_fragile_decision_probe_applies_materiality_and_concentration_thresholds():
    summary = phase3_macroplacement_ppo._summarize_fragile_decision_probe(
        per_step_results=[
            {"step_index": 1, "mean_hpwl_delta": 100_000.0, "cohort": "overlap"},
            {"step_index": 2, "mean_hpwl_delta": 2_000.0, "cohort": "overlap"},
            {"step_index": 3, "mean_hpwl_delta": 1_000.0, "cohort": "unique"},
            {"step_index": 4, "mean_hpwl_delta": 500.0, "cohort": "unique"},
            {"step_index": 5, "mean_hpwl_delta": 200.0, "cohort": "unique"},
        ],
        greedy_random_gap_hpwl=1_539_200.0,
    )

    assert summary["material_step_count"] == 1
    assert summary["concentration_band"] == "strong"
    assert summary["top_20pct_contribution_fraction"] > 0.8


def test_summarize_fragile_decision_probe_reports_init_rooted_when_cohort_medians_match():
    summary = phase3_macroplacement_ppo._summarize_fragile_decision_probe(
        per_step_results=[
            {"step_index": 1, "mean_hpwl_delta": 20_000.0, "cohort": "overlap"},
            {"step_index": 2, "mean_hpwl_delta": 18_000.0, "cohort": "overlap"},
            {"step_index": 3, "mean_hpwl_delta": 19_000.0, "cohort": "unique"},
            {"step_index": 4, "mean_hpwl_delta": 17_000.0, "cohort": "unique"},
        ],
        greedy_random_gap_hpwl=1_539_200.0,
    )

    assert summary["median_abs_delta_by_cohort"]["overlap"] == 19_000.0
    assert summary["median_abs_delta_by_cohort"]["unique"] == 18_000.0
    assert 0.5 <= summary["overlap_to_unique_median_ratio"] <= 2.0
    assert summary["init_rooted_band"] == "same_order_of_magnitude"


def test_run_ariane_fragile_decision_probe_records_priority_subset_and_panels(monkeypatch, tmp_path: Path):
    class _DummyPolicy:
        def __init__(self, label: str):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 2,
            "ordered_macro_count": 2,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0, "legality_penalty": 10.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), _DummyPolicy("warm_start"), reward_model, _StubAdapter(), metadata

    def fake_collect_reference_fixed_rollout(**kwargs):
        return {
            "reference_rollout": {
                "steps": 4,
                "episode_return": -123.0,
                "final_hpwl": 123.0,
                "termination_reason": "completed",
                "completed_episode": True,
            },
            "fixed_rollout": SimpleNamespace(transitions=[object(), object(), object(), object()]),
            "reference_trace": [
                {"step_index": 0, "macro": "m0", "target": (1.0, 1.0)},
                {"step_index": 1, "macro": "m1", "target": (2.0, 2.0)},
                {"step_index": 2, "macro": "m2", "target": (3.0, 3.0)},
                {"step_index": 3, "macro": "m3", "target": (4.0, 4.0)},
            ],
        }

    def fake_prefit_value_head_from_rollouts(**kwargs):
        return {"final_value_return_correlation": 0.8}

    def fake_run_single_ppo_update(**kwargs):
        kwargs["policy"].label = f"{kwargs['policy'].label}_updated"
        return {"substep_history": [{"substep_index": 0.0, "approx_kl": 0.4}], "final_hpwl": 111.0}

    def fake_collect_policy_rollout(**kwargs):
        return {
            "steps": 2,
            "invalid_action_count": 0,
            "transitions": [],
            "schema": "schema",
            "final_hpwl": 150.0,
            "episode_return": -150.0,
            "termination_reason": "completed",
            "completed_episode": True,
            "is_legal": True,
        }

    def fake_reference_step_metrics_for_fixed_rollout(**kwargs):
        return {
            "margin_by_step": {0: 0.20, 1: 0.01, 2: 0.03, 3: 0.04},
            "entropy_by_step": {0: 1.0, 1: 2.5, 2: 2.2, 3: 2.1},
        }

    def fake_extract_candidate_replacement_actions(**kwargs):
        policy = kwargs["candidate_policy"]
        if policy.label.startswith("warm_start_updated"):
            return {1: {"macro": "m1", "target": (20.0, 20.0)}, 2: {"macro": "m2", "target": (30.0, 30.0)}, 3: {"macro": "m3", "target": (40.0, 40.0)}}
        return {1: {"macro": "m1", "target": (21.0, 21.0)}, 3: {"macro": "m3", "target": (41.0, 41.0)}}

    def fake_run_single_step_counterfactual_panel(**kwargs):
        steps = kwargs["selected_steps"]
        label = kwargs["panel_label"]
        return {
            "panel_label": label,
            "step_count": len(steps),
            "per_step_results": [
                {
                    "step_index": int(step),
                    "mean_hpwl_delta": 20_000.0 if step == steps[0] else 1_000.0,
                    "cohort": "overlap" if int(step) in {1, 3} else "unique",
                }
                for step in steps
            ],
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_collect_ariane_fixed_reference_rollout", fake_collect_reference_fixed_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollouts", fake_prefit_value_head_from_rollouts)
    monkeypatch.setattr(phase3_macroplacement_ppo, "run_single_ppo_update", fake_run_single_ppo_update)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_reference_step_metrics_for_fixed_rollout", fake_reference_step_metrics_for_fixed_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_extract_candidate_replacement_actions", fake_extract_candidate_replacement_actions)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_run_single_step_counterfactual_panel", fake_run_single_step_counterfactual_panel)

    artifact = phase3_macroplacement_ppo.run_ariane_fragile_decision_probe(
        output_path=tmp_path / "phase3_ariane_fragile_decision_probe.json",
        baseline_flipped_step_indices=[1, 2, 3],
        alpha10_flipped_step_indices=[1, 3],
        suffix_seed_count=5,
        unique_budget=1,
    )

    assert artifact["priority_subset"]["overlap_steps"] == [1, 3]
    assert artifact["priority_subset"]["unique_steps"] == [2]
    assert artifact["probe_config"]["suffix_seed_count"] == 5
    assert artifact["panels"]["baseline"]["step_count"] == 3
    assert artifact["panels"]["alpha10_overlap"]["step_count"] == 2
    assert artifact["panel_summaries"]["baseline"]["material_step_count"] == 1


def test_run_single_step_counterfactual_panel_supports_deterministic_suffix_mode(monkeypatch):
    rollout_calls = []

    class _PanelAdapter:
        def ordered_macros(self):
            return ("m0", "m1", "m2")

        def initial_state(self):
            return {"actions": []}

        def apply_action(self, state, action):
            return {"actions": [*state["actions"], action]}

    def fake_collect_policy_rollout(**kwargs):
        rollout_calls.append(
            {
                "seed": kwargs["seed"],
                "deterministic": kwargs["deterministic"],
                "initial_state": kwargs["initial_state"],
            }
        )
        return {
            "final_hpwl": 100.0 + len(kwargs["initial_state"]["actions"]),
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)

    panel = phase3_macroplacement_ppo._run_single_step_counterfactual_panel(
        panel_label="screening",
        problem=SimpleNamespace(),
        reference_policy=SimpleNamespace(),
        adapter=_PanelAdapter(),
        reference_trace=[
            {"macro": "m0", "target": (1.0, 1.0)},
            {"macro": "m1", "target": (2.0, 2.0)},
        ],
        selected_steps=[1],
        replacement_action_by_step={1: {"macro": "m1", "target": (20.0, 20.0)}},
        cohort_by_step={1: "overlap"},
        suffix_seeds=(7,),
        grid_shape=(4, 4),
        adapter_kwargs={},
        deterministic_suffix=True,
    )

    assert panel["step_count"] == 1
    assert rollout_calls == [
        {
            "seed": 7,
            "deterministic": True,
            "initial_state": {
                "actions": [
                    {"macro": "m0", "target": (1.0, 1.0)},
                    {"macro": "m1", "target": (2.0, 2.0)},
                ]
            },
        },
        {
            "seed": 7,
            "deterministic": True,
            "initial_state": {
                "actions": [
                    {"macro": "m0", "target": (1.0, 1.0)},
                    {"macro": "m1", "target": (20.0, 20.0)},
                ]
            },
        },
    ]


def test_run_ariane_fragile_decision_screening_records_ranked_steps(monkeypatch, tmp_path: Path):
    class _DummyPolicy:
        def __init__(self, label: str):
            self.label = label

    def fake_build_components(*, artifact_dir, grid_shape=(32, 32), hidden_dim=64, reward_weights=None):
        del artifact_dir, reward_weights
        metadata = {
            "benchmark": "ariane133_ng45",
            "grid_shape": list(grid_shape),
            "hidden_dim": hidden_dim,
            "combined_lef_path": "combined.lef",
            "reference_def_path": "reference.def",
            "seed_macro_count": 2,
            "ordered_macro_count": 2,
            "reward_metadata": {"reward_weights": {"w_proxy": 0.0, "w_hpwl": 1.0, "legality_penalty": 10.0}},
        }
        reward_model = SimpleNamespace(hpwl_evaluator=object())
        return SimpleNamespace(), _DummyPolicy("warm_start"), reward_model, _StubAdapter(), metadata

    def fake_collect_reference_fixed_rollout(**kwargs):
        return {
            "reference_rollout": {
                "steps": 4,
                "episode_return": -123.0,
                "final_hpwl": 123.0,
                "termination_reason": "completed",
                "completed_episode": True,
            },
            "fixed_rollout": SimpleNamespace(transitions=[object(), object(), object(), object()]),
            "reference_trace": [
                {"step_index": 0, "macro": "m0", "target": (1.0, 1.0)},
                {"step_index": 1, "macro": "m1", "target": (2.0, 2.0)},
                {"step_index": 2, "macro": "m2", "target": (3.0, 3.0)},
                {"step_index": 3, "macro": "m3", "target": (4.0, 4.0)},
            ],
        }

    def fake_prefit_value_head_from_rollouts(**kwargs):
        return {"final_value_return_correlation": 0.8}

    def fake_run_single_ppo_update(**kwargs):
        kwargs["policy"].label = f"{kwargs['policy'].label}_updated"
        return {"substep_history": [{"substep_index": 0.0, "approx_kl": 0.4}], "final_hpwl": 111.0}

    def fake_collect_policy_rollout(**kwargs):
        return {
            "steps": 2,
            "invalid_action_count": 0,
            "transitions": [],
            "schema": "schema",
            "final_hpwl": 150.0,
            "episode_return": -150.0,
            "termination_reason": "completed",
            "completed_episode": True,
            "is_legal": True,
        }

    def fake_reference_step_metrics_for_fixed_rollout(**kwargs):
        return {
            "margin_by_step": {0: 0.20, 1: 0.01, 2: 0.03, 3: 0.04},
            "entropy_by_step": {0: 1.0, 1: 2.5, 2: 2.2, 3: 2.1},
        }

    def fake_extract_candidate_replacement_actions(**kwargs):
        policy = kwargs["candidate_policy"]
        if policy.label.startswith("warm_start_updated"):
            return {1: {"macro": "m1", "target": (20.0, 20.0)}, 2: {"macro": "m2", "target": (30.0, 30.0)}, 3: {"macro": "m3", "target": (40.0, 40.0)}}
        return {1: {"macro": "m1", "target": (21.0, 21.0)}, 3: {"macro": "m3", "target": (41.0, 41.0)}}

    def fake_run_single_step_counterfactual_panel(**kwargs):
        assert kwargs["deterministic_suffix"] is True
        steps = kwargs["selected_steps"]
        label = kwargs["panel_label"]
        return {
            "panel_label": label,
            "step_count": len(steps),
            "per_step_results": [
                {
                    "step_index": int(step),
                    "mean_hpwl_delta": {1: 20_000.0, 2: 5_000.0, 3: -30_000.0}[int(step)],
                    "cohort": "overlap" if int(step) in {1, 3} else "unique",
                }
                for step in steps
            ],
        }

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_collect_ariane_fixed_reference_rollout", fake_collect_reference_fixed_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "prefit_value_head_from_rollouts", fake_prefit_value_head_from_rollouts)
    monkeypatch.setattr(phase3_macroplacement_ppo, "run_single_ppo_update", fake_run_single_ppo_update)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_reference_step_metrics_for_fixed_rollout", fake_reference_step_metrics_for_fixed_rollout)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_extract_candidate_replacement_actions", fake_extract_candidate_replacement_actions)
    monkeypatch.setattr(phase3_macroplacement_ppo, "_run_single_step_counterfactual_panel", fake_run_single_step_counterfactual_panel)

    artifact = phase3_macroplacement_ppo.run_ariane_fragile_decision_screening(
        output_path=tmp_path / "phase3_ariane_fragile_decision_screening.json",
        baseline_flipped_step_indices=[1, 2, 3],
        alpha10_flipped_step_indices=[1, 3],
        unique_budget=1,
    )

    assert artifact["screening_config"]["deterministic_suffix"] is True
    assert artifact["panels"]["baseline"]["step_count"] == 3
    assert artifact["rankings"]["baseline_by_abs_delta"][0]["step_index"] == 3
    assert artifact["rankings"]["baseline_by_abs_delta"][0]["abs_mean_hpwl_delta"] == 30_000.0
    assert artifact["rankings"]["alpha10_overlap_by_abs_delta"][0]["step_index"] == 3
