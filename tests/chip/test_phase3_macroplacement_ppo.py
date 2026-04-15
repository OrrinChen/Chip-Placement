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
    assert artifact["metadata"]["training_config"]["init_seed"] == 0
    assert captured_train_kwargs["value_coef"] == 0.0
    assert captured_train_kwargs["entropy_coef"] == 5.0
    assert captured_train_kwargs["learning_rate"] == 1e-4
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
    assert artifact["trace_divergence"]["step_index"] == 1


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

    monkeypatch.setattr(phase3_macroplacement_ppo, "_build_ariane_components", fake_build_components)
    monkeypatch.setattr(phase3_macroplacement_ppo, "resolve_ariane_warm_start_checkpoint", lambda artifact_dir: tmp_path / "ariane_greedy_bc.pt")
    monkeypatch.setattr(phase3_macroplacement_ppo, "load_policy_warm_start", lambda *args, **kwargs: True)
    monkeypatch.setattr(phase3_macroplacement_ppo, "train_ppo_policy", fake_train_ppo_policy)
    monkeypatch.setattr(phase3_macroplacement_ppo, "collect_policy_rollout", fake_collect_policy_rollout)

    artifact = phase3_macroplacement_ppo.export_ariane_prefix_pool(
        output_path=tmp_path / "phase3_ariane_prefix_pool.json",
        num_updates=2,
        candidate_rollouts=2,
    )

    assert artifact["prefix_source"]["mode"] == "exported_policy_rollouts"
    assert artifact["base_final_rollout"]["final_hpwl"] == 2.9
    assert [row["action_indices"] for row in artifact["rollouts"]] == [[0, 1], [1, 2]]
    assert [row["seed"] for row in artifact["rollouts"]] == [0, 1]


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
        geometry_packing_factor=1.0,
        min_accepted_prefixes=3,
    )

    assert train_calls == [0, 1]
    assert artifact["prefix_source"]["mode"] == "exported_policy_rollouts"
    assert artifact["prefix_source"]["policy_seed_count"] == 2
    assert artifact["acceptance_criteria"]["hpwl_threshold"] == 2.85
    assert artifact["acceptance_summary"]["total_rollouts"] == 4
    assert artifact["acceptance_summary"]["accepted_prefixes"] == 3
    assert artifact["acceptance_summary"]["meets_min_prefix_count"] is True
    assert artifact["acceptance_summary"]["status"] == "pass"
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
