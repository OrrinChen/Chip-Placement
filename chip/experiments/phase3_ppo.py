from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from chip.experiments.phase3_imitation import FeatureSchema
from chip.experiments.phase3_imitation import encode_problem_graph
from chip.experiments.phase3_imitation import fit_feature_schema
from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from engine.gnn_proxy import GraphBatch
from engine.gnn_proxy import batch_graph_tensors
from engine.ppo_policy import PPOPlacementPolicy
from engine.rl_env import PlacementEnv


@dataclass(frozen=True)
class PolicyInputBatch:
    graph_batch: GraphBatch
    canvas: torch.Tensor
    current_macro_index: torch.Tensor
    action_mask: torch.Tensor


@dataclass(frozen=True)
class RolloutStep:
    observation: dict[str, object]
    action_index: int
    log_prob: float
    reward: float
    value: float


@dataclass(frozen=True)
class FixedRolloutBatch:
    transitions: list[RolloutStep]
    schema: FeatureSchema
    final_hpwl: float
    episode_return: float
    termination_reason: str
    completed_episode: bool


def initialize_policy_for_problem(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    hidden_dim: int = 64,
) -> PPOPlacementPolicy:
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    observation = adapter.observe(adapter.initial_state(), step_index=0)
    schema = fit_feature_schema([observation["graph"]])
    encoded_graph = encode_problem_graph(observation["graph"], schema)
    return PPOPlacementPolicy(
        node_feature_dim=encoded_graph["node_features"].shape[1],
        graph_feature_dim=encoded_graph["graph_features"].shape[0],
        grid_channels=int(observation["canvas"].shape[0]),
        grid_shape=grid_shape,
        hidden_dim=hidden_dim,
        max_macro_index=max(len(adapter.ordered_macros()) + 1, 8),
    )


def load_policy_warm_start(policy, checkpoint_path: Path) -> bool:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key: value
        for key, value in checkpoint["policy_state_dict"].items()
        if not key.startswith("value_head.")
    }
    policy.load_state_dict(state_dict, strict=False)
    return True


def collect_policy_rollout(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    max_steps: int = 1,
    seed: int = 0,
    deterministic: bool = False,
    uniform_legal_policy: bool = False,
    adapter_kwargs: dict[str, object] | None = None,
    initial_state=None,
    policy_seed: int | None = None,
) -> dict[str, object]:
    del policy_seed
    torch.manual_seed(seed)
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape, **(adapter_kwargs or {}))
    env = PlacementEnv(adapter=adapter, max_steps=max_steps, reward_source=reward_source)
    reset_options = None
    if initial_state is not None:
        reset_options = {
            "initial_state": initial_state,
            "initial_step_index": len(getattr(initial_state, "placed_macros", ())),
        }
    observation, _ = env.reset(seed=seed, options=reset_options)
    schema = fit_feature_schema([observation["graph"]])
    steps: list[RolloutStep] = []
    invalid_action_count = 0
    episode_return = 0.0
    termination_reason = "completed"

    while True:
        if observation["current_macro"] is None:
            termination_reason = "completed"
            break
        current_mask = observation["action_mask"]
        if bool(current_mask.all()):
            termination_reason = "no_valid_actions"
            break
        batch = build_policy_input_batch([observation], schema=schema)
        with torch.no_grad():
            logits, value = policy(
                graph_batch=batch.graph_batch,
                canvas=batch.canvas,
                current_macro_index=batch.current_macro_index,
                action_mask=batch.action_mask,
            )
        policy_logits = logits
        if uniform_legal_policy:
            policy_logits = torch.zeros_like(logits).masked_fill(batch.action_mask.view(batch.action_mask.shape[0], -1), float("-inf"))
        action_distribution = torch.distributions.Categorical(logits=policy_logits)
        if deterministic:
            action_index = int(policy_logits.argmax(dim=1).item())
            log_prob = float(action_distribution.log_prob(torch.tensor([action_index])).item())
        else:
            action_index = int(action_distribution.sample().item())
            log_prob = float(action_distribution.log_prob(torch.tensor([action_index])).item())
        if bool(batch.action_mask.view(1, -1)[0, action_index]):
            invalid_action_count += 1

        current_macro = observation["current_macro"]
        row_index = action_index // grid_shape[1]
        col_index = action_index % grid_shape[1]
        target = adapter.grid_index_to_coordinate((row_index, col_index), grid_shape)
        action = {"macro": current_macro, "target": target}
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        steps.append(
            RolloutStep(
                observation=observation,
                action_index=action_index,
                log_prob=log_prob,
                reward=float(reward),
                value=float(value.squeeze(0).item()),
            )
        )
        observation = next_observation
        if terminated or truncated:
            termination_reason = "completed" if terminated else "truncated"
            break

    return {
        "steps": len(steps),
        "invalid_action_count": invalid_action_count,
        "transitions": steps,
        "schema": schema,
        "final_hpwl": adapter.current_hpwl(env.state),
        "episode_return": float(episode_return),
        "termination_reason": termination_reason,
        "completed_episode": termination_reason == "completed",
        "is_legal": adapter.constraint_report(env.state).is_valid,
    }


def _mean_rollout_action_log_prob(
    *,
    policy: PPOPlacementPolicy,
    rollout: dict[str, object],
) -> float:
    transitions: list[RolloutStep] = rollout.get("transitions", [])
    if not transitions:
        return 0.0
    schema: FeatureSchema = rollout["schema"]
    batch = build_policy_input_batch([step.observation for step in transitions], schema=schema)
    action_index = torch.tensor([step.action_index for step in transitions], dtype=torch.long)
    with torch.no_grad():
        logits, _ = policy(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
    distribution = torch.distributions.Categorical(logits=logits)
    return float(distribution.log_prob(action_index).mean().item())


def _summarize_scalar_tensor(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().reshape(-1).to(torch.float32)
    if flat.numel() == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=flat.dtype, device=flat.device),
    )
    return {
        "count": float(flat.numel()),
        "min": float(flat.min().item()),
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "max": float(flat.max().item()),
    }


def _summarize_named_histogram(
    values: torch.Tensor,
    *,
    buckets: list[tuple[str, float | None, float | None]],
) -> dict[str, dict[str, float]]:
    flat = values.detach().reshape(-1).to(torch.float32)
    total = int(flat.numel())
    summary: dict[str, dict[str, float]] = {}
    for label, lower, upper in buckets:
        mask = torch.ones_like(flat, dtype=torch.bool)
        if lower is not None:
            mask = mask & (flat >= float(lower))
        if upper is not None:
            mask = mask & (flat < float(upper))
        count = int(mask.sum().item()) if total > 0 else 0
        summary[label] = {
            "count": count,
            "fraction": float(count / total) if total > 0 else 0.0,
        }
    return summary


def _summarize_agreement_bucket(
    *,
    indices: list[int],
    agreements: torch.Tensor,
    areas: torch.Tensor | None = None,
) -> dict[str, float]:
    if not indices:
        return {
            "step_count": 0,
            "agreement_count": 0,
            "flip_count": 0,
            "agreement_rate": 0.0,
            "mean_macro_area": 0.0 if areas is not None else None,
        }
    index_tensor = torch.tensor(indices, dtype=torch.long)
    bucket_agreements = agreements[index_tensor]
    agreement_count = int(bucket_agreements.sum().item())
    step_count = int(index_tensor.numel())
    flip_count = int(step_count - agreement_count)
    mean_macro_area = None
    if areas is not None:
        mean_macro_area = float(areas[index_tensor].mean().item())
    return {
        "step_count": step_count,
        "agreement_count": agreement_count,
        "flip_count": flip_count,
        "agreement_rate": float(agreement_count / max(1, step_count)),
        "mean_macro_area": mean_macro_area,
    }


def _top_abs_advantages(
    *,
    raw_advantages: torch.Tensor,
    normalized_advantages: torch.Tensor,
    returns: torch.Tensor,
    old_value: torch.Tensor,
    rewards: list[float],
    limit: int = 5,
) -> list[dict[str, float]]:
    if raw_advantages.numel() == 0:
        return []
    indices = torch.argsort(raw_advantages.abs(), descending=True)[: max(1, int(limit))]
    rows: list[dict[str, float]] = []
    for index in indices.tolist():
        rows.append(
            {
                "transition_index": float(index),
                "reward": float(rewards[index]),
                "return": float(returns[index].item()),
                "old_value": float(old_value[index].item()),
                "raw_advantage": float(raw_advantages[index].item()),
                "normalized_advantage": float(normalized_advantages[index].item()),
            }
        )
    return rows


def _safe_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum()))
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float((x_centered * y_centered).sum().div(denominator).item())


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().to(torch.float32).square().sum().item())
    return float(total ** 0.5)


def _global_grad_norm(policy: PPOPlacementPolicy) -> float:
    total = 0.0
    for parameter in policy.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().to(torch.float32).square().sum().item())
    return float(total ** 0.5)


def _parameter_delta_norm(
    policy: PPOPlacementPolicy,
    reference_parameters: dict[str, torch.Tensor],
    *,
    name_prefix: str | None = None,
) -> float:
    total = 0.0
    for name, parameter in policy.named_parameters():
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        reference = reference_parameters[name].to(device=parameter.device, dtype=parameter.dtype)
        delta = parameter.detach() - reference
        total += float(delta.to(torch.float32).square().sum().item())
    return float(total ** 0.5)


def _advantage_gate(
    raw_advantages: torch.Tensor,
    *,
    std_eps: float = 1e-6,
    mean_over_std_limit: float = 10.0,
) -> dict[str, float | bool]:
    flat = raw_advantages.detach().reshape(-1).to(torch.float32)
    if flat.numel() == 0:
        return {
            "std_epsilon": float(std_eps),
            "mean_over_std_limit": float(mean_over_std_limit),
            "mean": 0.0,
            "std": 0.0,
            "mean_over_std": 0.0,
            "std_gt_epsilon": False,
            "mean_over_std_within_band": False,
            "pass": False,
        }
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    std_gt_epsilon = torch.isfinite(flat).all().item() and std > float(std_eps)
    mean_over_std = abs(mean) / max(std, float(std_eps))
    mean_over_std_within_band = bool(std_gt_epsilon and mean_over_std < float(mean_over_std_limit))
    return {
        "std_epsilon": float(std_eps),
        "mean_over_std_limit": float(mean_over_std_limit),
        "mean": mean,
        "std": std,
        "mean_over_std": float(mean_over_std),
        "std_gt_epsilon": bool(std_gt_epsilon),
        "mean_over_std_within_band": bool(mean_over_std_within_band),
        "pass": bool(std_gt_epsilon and mean_over_std_within_band),
    }


def _value_target_stats_snapshot(policy: PPOPlacementPolicy) -> dict[str, float]:
    return {
        "mean": float(policy.value_target_mean.detach().to(torch.float32).item()),
        "std": float(policy.value_target_std.detach().to(torch.float32).item()),
    }


def _ratio_tail_metrics(
    *,
    ratio: torch.Tensor,
    kl_terms: torch.Tensor,
    high_threshold: float = 1.5,
    low_threshold: float = 0.67,
) -> dict[str, float]:
    flat_ratio = ratio.detach().reshape(-1).to(torch.float32)
    flat_kl = kl_terms.detach().reshape(-1).to(torch.float32).abs()
    if flat_ratio.numel() == 0:
        return {
            "ratio_fraction_gt_1_5": 0.0,
            "ratio_fraction_lt_0_67": 0.0,
            "ratio_max": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_p90": 0.0,
            "approx_kl_abs_top10pct_fraction": 0.0,
        }
    top_count = max(1, int(torch.ceil(torch.tensor(flat_kl.numel() * 0.1)).item()))
    top_abs_kl = torch.topk(flat_kl, k=min(top_count, flat_kl.numel())).values.sum()
    total_abs_kl = flat_kl.sum()
    top_fraction = 0.0
    if float(total_abs_kl.item()) > 1e-12:
        top_fraction = float((top_abs_kl / total_abs_kl).item())
    quantiles = torch.quantile(
        flat_ratio,
        torch.tensor([0.90], dtype=flat_ratio.dtype, device=flat_ratio.device),
    )
    return {
        "ratio_fraction_gt_1_5": float((flat_ratio > float(high_threshold)).to(torch.float32).mean().item()),
        "ratio_fraction_lt_0_67": float((flat_ratio < float(low_threshold)).to(torch.float32).mean().item()),
        "ratio_max": float(flat_ratio.max().item()),
        "ratio_mean": float(flat_ratio.mean().item()),
        "ratio_std": float(flat_ratio.std(unbiased=False).item()),
        "ratio_p90": float(quantiles[0].item()),
        "approx_kl_abs_top10pct_fraction": top_fraction,
    }


def run_single_ppo_update(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
    reference_policy: PPOPlacementPolicy | None = None,
    kl_anchor_beta: float = 0.0,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    max_steps: int = 1,
    learning_rate: float = 1e-3,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 0,
    adapter_kwargs: dict[str, object] | None = None,
    uniform_legal_policy: bool = False,
    initial_state=None,
    policy_epochs: int = 1,
    diagnostic_eval_seeds: tuple[int, ...] | None = None,
    grad_clip_norm: float | None = None,
    first_substep_lr_scale: float | None = None,
) -> dict[str, object]:
    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=grid_shape,
        reward_source=reward_source,
        max_steps=max_steps,
        seed=seed,
        adapter_kwargs=adapter_kwargs,
        uniform_legal_policy=uniform_legal_policy,
        initial_state=initial_state,
    )
    transitions: list[RolloutStep] = rollout["transitions"]
    schema: FeatureSchema = rollout["schema"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }

    returns = compute_discounted_returns([step.reward for step in transitions], gamma=gamma)
    batch = build_policy_input_batch([step.observation for step in transitions], schema=schema)
    action_index = torch.tensor([step.action_index for step in transitions], dtype=torch.long)
    old_log_prob = torch.tensor([step.log_prob for step in transitions], dtype=torch.float32)
    old_value = torch.tensor([step.value for step in transitions], dtype=torch.float32)
    raw_advantages = returns - old_value
    advantages = normalize_advantages(raw_advantages)
    batch_diagnostic = {
        "return_summary": _summarize_scalar_tensor(returns),
        "old_value_summary": _summarize_scalar_tensor(old_value),
        "raw_advantage_summary": _summarize_scalar_tensor(raw_advantages),
        "normalized_advantage_summary": _summarize_scalar_tensor(advantages),
        "top_abs_advantages": _top_abs_advantages(
            raw_advantages=raw_advantages,
            normalized_advantages=advantages,
            returns=returns,
            old_value=old_value,
            rewards=[float(step.reward) for step in transitions],
        ),
        "value_return_correlation": _safe_correlation(old_value, returns),
        "advantage_gate": _advantage_gate(raw_advantages),
    }

    reference_distribution = None
    if reference_policy is not None and float(kl_anchor_beta) > 0.0:
        with torch.no_grad():
            reference_logits, _ = reference_policy(
                graph_batch=batch.graph_batch,
                canvas=batch.canvas,
                current_macro_index=batch.current_macro_index,
                action_mask=batch.action_mask,
            )
        reference_distribution = torch.distributions.Categorical(logits=reference_logits)

    heldout_eval_seeds = tuple(int(current_seed) for current_seed in (diagnostic_eval_seeds or ()))
    heldout_reference_policy = reference_policy if reference_policy is not None else policy
    heldout_reference_rollouts = [
        collect_policy_rollout(
            problem=problem,
            policy=heldout_reference_policy,
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max_steps,
            seed=current_seed,
            adapter_kwargs=adapter_kwargs,
            uniform_legal_policy=uniform_legal_policy,
            initial_state=initial_state,
        )
        for current_seed in heldout_eval_seeds
    ]

    def _compute_current_metrics() -> dict[str, torch.Tensor]:
        logits, value_train_space = policy.forward_train_space(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
        value_raw = policy.decode_value_predictions(value_train_space).squeeze(-1)
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_prob = distribution.log_prob(action_index)
        entropy = distribution.entropy().mean()
        kl_anchor_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if reference_distribution is not None:
            kl_anchor_loss = torch.distributions.kl.kl_divergence(reference_distribution, distribution).mean()
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_targets_train_space = policy.encode_value_targets(returns)
        value_loss_train_space = torch.nn.functional.mse_loss(value_train_space.squeeze(-1), value_targets_train_space)
        value_loss_raw_space = torch.nn.functional.mse_loss(value_raw, returns)
        total_loss = policy_loss + (value_coef * value_loss_train_space) - (entropy_coef * entropy) + (float(kl_anchor_beta) * kl_anchor_loss)
        kl_terms = old_log_prob - new_log_prob
        approx_kl = kl_terms.mean()
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss_train_space,
            "value_loss_train_space": value_loss_train_space,
            "value_loss_raw_space": value_loss_raw_space,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "kl_anchor_loss": kl_anchor_loss,
            "total_loss": total_loss,
            "current_value_summary_raw": _summarize_scalar_tensor(value_raw),
            "current_value_return_correlation": torch.tensor(
                _safe_correlation(value_raw, returns),
                dtype=value_raw.dtype,
                device=value_raw.device,
            ),
            "ratio_tail_metrics": _ratio_tail_metrics(ratio=ratio, kl_terms=kl_terms),
        }

    def _collect_diagnostic_snapshot(
        *,
        update_batch_kl_to_reference: float,
    ) -> dict[str, float] | None:
        if not heldout_eval_seeds:
            return None
        eval_rollouts = [
            collect_policy_rollout(
                problem=problem,
                policy=policy,
                grid_shape=grid_shape,
                reward_source=reward_source,
                max_steps=max_steps,
                seed=current_seed,
                adapter_kwargs=adapter_kwargs,
                uniform_legal_policy=uniform_legal_policy,
                initial_state=initial_state,
            )
            for current_seed in heldout_eval_seeds
        ]
        completion_count = sum(1 for eval_rollout in eval_rollouts if bool(eval_rollout["completed_episode"]))
        hpwls = [float(eval_rollout["final_hpwl"]) for eval_rollout in eval_rollouts]
        log_prob_deltas: list[float] = []
        for reference_rollout in heldout_reference_rollouts:
            reference_transitions: list[RolloutStep] = reference_rollout.get("transitions", [])
            if not reference_transitions:
                continue
            reference_mean_log_prob = sum(float(step.log_prob) for step in reference_transitions) / len(reference_transitions)
            current_mean_log_prob = _mean_rollout_action_log_prob(policy=policy, rollout=reference_rollout)
            log_prob_deltas.append(float(current_mean_log_prob - reference_mean_log_prob))
        return {
            "heldout_rollout_count": float(len(eval_rollouts)),
            "heldout_completion_rate": float(completion_count / max(1, len(eval_rollouts))),
            "heldout_mean_hpwl": float(sum(hpwls) / max(1, len(hpwls))),
            "heldout_log_prob_delta_mean": float(sum(log_prob_deltas) / max(1, len(log_prob_deltas))),
            "update_batch_kl_to_reference": float(update_batch_kl_to_reference),
        }

    pre_update_metrics = _compute_current_metrics()
    pre_update_diagnostic = _collect_diagnostic_snapshot(
        update_batch_kl_to_reference=float(pre_update_metrics["kl_anchor_loss"].item())
    )
    pre_update_metrics_summary = {
        "policy_loss": float(pre_update_metrics["policy_loss"].item()),
        "value_loss": float(pre_update_metrics["value_loss"].item()),
        "value_loss_train_space": float(pre_update_metrics["value_loss_train_space"].item()),
        "value_loss_raw_space": float(pre_update_metrics["value_loss_raw_space"].item()),
        "entropy": float(pre_update_metrics["entropy"].item()),
        "approx_kl": float(pre_update_metrics["approx_kl"].item()),
        "kl_anchor_loss": float(pre_update_metrics["kl_anchor_loss"].item()),
        "current_value_mean_raw": float(pre_update_metrics["current_value_summary_raw"]["mean"]),
        "current_value_std_raw": float(pre_update_metrics["current_value_summary_raw"]["std"]),
        "current_value_return_correlation": float(pre_update_metrics["current_value_return_correlation"].item()),
        **pre_update_metrics["ratio_tail_metrics"],
    }

    substep_history: list[dict[str, float]] = []
    final_metrics = pre_update_metrics
    for substep_index in range(max(1, int(policy_epochs))):
        effective_learning_rate = float(learning_rate)
        if substep_index == 0 and first_substep_lr_scale is not None:
            effective_learning_rate = float(learning_rate) * float(first_substep_lr_scale)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = effective_learning_rate
        current_metrics = _compute_current_metrics()
        optimizer.zero_grad()
        current_metrics["total_loss"].backward()
        global_grad_norm = _global_grad_norm(policy)
        policy_head_grad_norm = _grad_norm(policy.policy_head)
        value_head_grad_norm = _grad_norm(policy.value_head)
        post_clip_global_grad_norm = float(global_grad_norm)
        grad_clip_applied = False
        clip_value = None if grad_clip_norm is None else float(grad_clip_norm)
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_value)
            post_clip_global_grad_norm = _global_grad_norm(policy)
            grad_clip_applied = bool(global_grad_norm > clip_value + 1e-12)
        optimizer.step()
        final_metrics = _compute_current_metrics()
        global_param_delta_norm = _parameter_delta_norm(policy, initial_parameters)
        policy_head_param_delta_norm = _parameter_delta_norm(
            policy,
            initial_parameters,
            name_prefix="policy_head.",
        )
        substep_row: dict[str, float] = {
            "substep_index": float(substep_index),
            "policy_loss": float(final_metrics["policy_loss"].item()),
            "value_loss": float(final_metrics["value_loss"].item()),
            "value_loss_train_space": float(final_metrics["value_loss_train_space"].item()),
            "value_loss_raw_space": float(final_metrics["value_loss_raw_space"].item()),
            "entropy": float(final_metrics["entropy"].item()),
            "approx_kl": float(final_metrics["approx_kl"].item()),
            "update_batch_kl_to_reference": float(final_metrics["kl_anchor_loss"].item()),
            "global_grad_norm": float(global_grad_norm),
            "global_grad_norm_pre_clip": float(global_grad_norm),
            "global_grad_norm_post_clip": float(post_clip_global_grad_norm),
            "policy_head_grad_norm": float(policy_head_grad_norm),
            "value_head_grad_norm": float(value_head_grad_norm),
            "current_value_mean_raw": float(final_metrics["current_value_summary_raw"]["mean"]),
            "current_value_std_raw": float(final_metrics["current_value_summary_raw"]["std"]),
            "current_value_return_correlation": float(final_metrics["current_value_return_correlation"].item()),
            "value_target_mean": float(policy.value_target_mean.detach().to(torch.float32).item()),
            "value_target_std": float(policy.value_target_std.detach().to(torch.float32).item()),
            "entropy_delta_from_pre_update": float(final_metrics["entropy"].item() - pre_update_metrics["entropy"].item()),
            "grad_clip_norm": clip_value,
            "grad_clip_applied": bool(grad_clip_applied),
            "effective_learning_rate": float(effective_learning_rate),
            "first_substep_lr_scale": None if first_substep_lr_scale is None else float(first_substep_lr_scale),
            "global_param_delta_norm_from_pre_update": float(global_param_delta_norm),
            "policy_head_param_delta_norm_from_pre_update": float(policy_head_param_delta_norm),
        }
        substep_row.update(final_metrics["ratio_tail_metrics"])
        diagnostic_snapshot = _collect_diagnostic_snapshot(
            update_batch_kl_to_reference=float(final_metrics["kl_anchor_loss"].item())
        )
        if diagnostic_snapshot is not None:
            substep_row.update(diagnostic_snapshot)
        substep_history.append(substep_row)

    return {
        "policy_loss": float(final_metrics["policy_loss"].item()),
        "value_loss": float(final_metrics["value_loss"].item()),
        "value_loss_train_space": float(final_metrics["value_loss_train_space"].item()),
        "value_loss_raw_space": float(final_metrics["value_loss_raw_space"].item()),
        "entropy": float(final_metrics["entropy"].item()),
        "steps": float(len(transitions)),
        "final_hpwl": float(rollout["final_hpwl"]),
        "invalid_action_count": float(rollout["invalid_action_count"]),
        "episode_return": float(rollout["episode_return"]),
        "completed_episode": 1.0 if rollout["completed_episode"] else 0.0,
        "is_legal": 1.0 if rollout["is_legal"] else 0.0,
        "approx_kl": float(final_metrics["approx_kl"].item()),
        "kl_anchor_loss": float(final_metrics["kl_anchor_loss"].item()),
        "kl_anchor_beta": float(kl_anchor_beta),
        "policy_epochs": float(max(1, int(policy_epochs))),
        "pre_update_diagnostic": pre_update_diagnostic,
        "pre_update_metrics": pre_update_metrics_summary,
        "batch_diagnostic": batch_diagnostic,
        "value_target_stats": _value_target_stats_snapshot(policy),
        "substep_history": substep_history,
    }


def train_ppo_policy(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
    reference_policy: PPOPlacementPolicy | None = None,
    kl_anchor_beta: float = 0.0,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    max_steps: int = 1,
    num_updates: int = 1,
    learning_rate: float = 1e-3,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 0,
    adapter_kwargs: dict[str, object] | None = None,
    uniform_legal_policy: bool = False,
    progress_every: int | None = None,
    initial_state_factory=None,
    policy_epochs: int = 1,
    diagnostic_eval_seeds: tuple[int, ...] | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, object]:
    history: list[dict[str, float]] = []
    best_hpwl_so_far: float | None = None
    for update_index in range(max(1, num_updates)):
        update_metrics = run_single_ppo_update(
            problem=problem,
            policy=policy,
            reference_policy=reference_policy,
            kl_anchor_beta=kl_anchor_beta,
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max_steps,
            learning_rate=learning_rate,
            clip_epsilon=clip_epsilon,
            gamma=gamma,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            seed=seed + update_index,
            adapter_kwargs=adapter_kwargs,
            uniform_legal_policy=uniform_legal_policy,
            initial_state=None if initial_state_factory is None else initial_state_factory(update_index),
            policy_epochs=policy_epochs,
            diagnostic_eval_seeds=diagnostic_eval_seeds,
            grad_clip_norm=grad_clip_norm,
        )
        current_hpwl = float(update_metrics["final_hpwl"])
        best_hpwl_so_far = current_hpwl if best_hpwl_so_far is None else min(best_hpwl_so_far, current_hpwl)
        history.append(
            {
                **update_metrics,
                "best_hpwl_so_far": float(best_hpwl_so_far),
            }
        )
        if progress_every is not None and progress_every > 0:
            if (update_index + 1) % progress_every == 0 or update_index == 0 or update_index == num_updates - 1:
                print(
                    {
                        "update": update_index + 1,
                        "final_hpwl": current_hpwl,
                        "episode_return": float(update_metrics["episode_return"]),
                        "entropy": float(update_metrics["entropy"]),
                        "approx_kl": float(update_metrics["approx_kl"]),
                        "completed_episode": bool(update_metrics["completed_episode"]),
                    },
                    flush=True,
                )

    final_rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=grid_shape,
        reward_source=reward_source,
        max_steps=max_steps,
        seed=seed,
        deterministic=True,
        adapter_kwargs=adapter_kwargs,
    )
    return {
        "history": history,
        "final_rollout": final_rollout,
    }


def collect_rollout_for_action_trace(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
    action_trace: list[dict[str, object]],
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    max_steps: int | None = None,
    seed: int = 0,
    adapter_kwargs: dict[str, object] | None = None,
) -> FixedRolloutBatch:
    torch.manual_seed(seed)
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape, **(adapter_kwargs or {}))
    env = PlacementEnv(
        adapter=adapter,
        max_steps=max_steps if max_steps is not None else max(1, len(action_trace)),
        reward_source=reward_source,
    )
    observation, _ = env.reset(seed=seed)
    schema = fit_feature_schema([observation["graph"]])
    transitions: list[RolloutStep] = []
    episode_return = 0.0
    termination_reason = "completed"

    for trace_row in action_trace:
        if observation["current_macro"] is None:
            termination_reason = "completed"
            break
        action_mask = observation["action_mask"]
        if bool(action_mask.all()):
            termination_reason = "no_valid_actions"
            break
        action = dict(trace_row["action"]) if "action" in trace_row else dict(trace_row)
        row_index, col_index = adapter.coordinate_to_grid_index(tuple(action["target"]), grid_shape)
        action_index = row_index * grid_shape[1] + col_index
        batch = build_policy_input_batch([observation], schema=schema)
        with torch.no_grad():
            logits, value = policy(
                graph_batch=batch.graph_batch,
                canvas=batch.canvas,
                current_macro_index=batch.current_macro_index,
                action_mask=batch.action_mask,
            )
        distribution = torch.distributions.Categorical(logits=logits)
        log_prob = float(distribution.log_prob(torch.tensor([action_index])).item())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        transitions.append(
            RolloutStep(
                observation=observation,
                action_index=action_index,
                log_prob=log_prob,
                reward=float(reward),
                value=float(value.squeeze(0).item()),
            )
        )
        observation = next_observation
        if terminated or truncated:
            termination_reason = "completed" if terminated else "truncated"
            break
    else:
        current_step = len(transitions)
        next_observation = adapter.observe(env.state, current_step)
        if next_observation["current_macro"] is None:
            termination_reason = "completed"
        elif bool(next_observation["action_mask"].all()):
            termination_reason = "no_valid_actions"
        elif current_step >= env.max_steps:
            termination_reason = "truncated"
        else:
            termination_reason = "trace_exhausted"

    return FixedRolloutBatch(
        transitions=transitions,
        schema=schema,
        final_hpwl=float(adapter.current_hpwl(env.state)),
        episode_return=float(episode_return),
        termination_reason=termination_reason,
        completed_episode=termination_reason == "completed",
    )


def summarize_argmax_agreement_for_fixed_rollout(
    *,
    reference_policy,
    candidate_policy,
    fixed_rollout: FixedRolloutBatch,
) -> dict[str, object]:
    observations = [step.observation for step in fixed_rollout.transitions]
    if not observations:
        empty_summary = _summarize_scalar_tensor(torch.empty((0,), dtype=torch.float32))
        return {
            "step_count": 0,
            "agreement_count": 0,
            "flip_count": 0,
            "overall_agreement_rate": 0.0,
            "flipped_step_indices": [],
            "reference_argmax_indices": [],
            "candidate_argmax_indices": [],
            "pre_update_entropy_summary": empty_summary,
            "pre_update_margin_summary": empty_summary,
            "flipped_entropy_summary": empty_summary,
            "non_flipped_entropy_summary": empty_summary,
            "flipped_margin_summary": empty_summary,
            "non_flipped_margin_summary": empty_summary,
            "flipped_entropy_histogram": {},
            "non_flipped_entropy_histogram": {},
            "flipped_margin_histogram": {},
            "non_flipped_margin_histogram": {},
            "position_buckets": {
                name: {
                    "step_count": 0,
                    "agreement_count": 0,
                    "flip_count": 0,
                    "agreement_rate": 0.0,
                    "mean_macro_area": None,
                }
                for name in ("early", "mid", "late")
            },
            "macro_area_buckets": {
                name: {
                    "step_count": 0,
                    "agreement_count": 0,
                    "flip_count": 0,
                    "agreement_rate": 0.0,
                    "mean_macro_area": None,
                }
                for name in ("small", "medium", "large")
            },
        }

    batch = build_policy_input_batch(observations, schema=fixed_rollout.schema)
    with torch.no_grad():
        reference_logits, _ = reference_policy(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
        candidate_logits, _ = candidate_policy(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )

    reference_argmax = reference_logits.argmax(dim=1)
    candidate_argmax = candidate_logits.argmax(dim=1)
    agreements = reference_argmax.eq(candidate_argmax)
    flipped_indices = (~agreements).nonzero(as_tuple=False).view(-1)

    reference_distribution = torch.distributions.Categorical(logits=reference_logits)
    pre_update_entropy = reference_distribution.entropy().to(torch.float32)
    reference_probabilities = torch.softmax(reference_logits, dim=1)
    topk_count = 2 if reference_probabilities.shape[1] >= 2 else 1
    top_probabilities = torch.topk(reference_probabilities, k=topk_count, dim=1).values
    if topk_count == 1:
        pre_update_margin = top_probabilities[:, 0]
    else:
        pre_update_margin = top_probabilities[:, 0] - top_probabilities[:, 1]
    macro_areas = []
    for observation in observations:
        current_macro = str(observation["current_macro"])
        graph = observation["graph"]
        node_payload = graph.nodes[current_macro]
        macro_areas.append(float(node_payload["width"]) * float(node_payload["height"]))
    macro_area_tensor = torch.tensor(macro_areas, dtype=torch.float32)

    def _bucket_name(step_index: int, total_steps: int) -> str:
        if total_steps <= 1:
            return "early"
        normalized_position = float(step_index) / float(total_steps - 1)
        if normalized_position <= (1.0 / 3.0):
            return "early"
        if normalized_position < (2.0 / 3.0):
            return "mid"
        return "late"

    bucket_rows: dict[str, list[int]] = {"early": [], "mid": [], "late": []}
    for step_index in range(len(observations)):
        bucket_rows[_bucket_name(step_index, len(observations))].append(step_index)

    position_buckets: dict[str, dict[str, float]] = {}
    for bucket_name, indices in bucket_rows.items():
        position_buckets[bucket_name] = _summarize_agreement_bucket(
            indices=indices,
            agreements=agreements,
            areas=macro_area_tensor,
        )

    macro_area_buckets: dict[str, list[int]] = {"small": [], "medium": [], "large": []}
    if macro_area_tensor.numel() > 0:
        q33 = float(torch.quantile(macro_area_tensor, torch.tensor(1.0 / 3.0)).item())
        q67 = float(torch.quantile(macro_area_tensor, torch.tensor(2.0 / 3.0)).item())
        for step_index, area in enumerate(macro_areas):
            if abs(q67 - q33) <= 1e-8:
                bucket_name = "medium"
            elif area <= q33:
                bucket_name = "small"
            elif area < q67:
                bucket_name = "medium"
            else:
                bucket_name = "large"
            macro_area_buckets[bucket_name].append(step_index)
    macro_area_bucket_summary = {
        bucket_name: _summarize_agreement_bucket(
            indices=indices,
            agreements=agreements,
            areas=macro_area_tensor,
        )
        for bucket_name, indices in macro_area_buckets.items()
    }

    entropy_histogram_buckets = [
        ("lt_0_50", None, 0.5),
        ("0_50_to_1_00", 0.5, 1.0),
        ("1_00_to_1_50", 1.0, 1.5),
        ("1_50_to_2_00", 1.5, 2.0),
        ("gte_2_00", 2.0, None),
    ]
    margin_histogram_buckets = [
        ("lt_0_05", None, 0.05),
        ("0_05_to_0_10", 0.05, 0.10),
        ("0_10_to_0_20", 0.10, 0.20),
        ("0_20_to_0_50", 0.20, 0.50),
        ("gte_0_50", 0.50, None),
    ]

    return {
        "step_count": int(len(observations)),
        "agreement_count": int(agreements.sum().item()),
        "flip_count": int((~agreements).sum().item()),
        "overall_agreement_rate": float(agreements.to(torch.float32).mean().item()),
        "flipped_step_indices": [int(index) for index in flipped_indices.tolist()],
        "reference_argmax_indices": [int(index) for index in reference_argmax.tolist()],
        "candidate_argmax_indices": [int(index) for index in candidate_argmax.tolist()],
        "pre_update_entropy_summary": _summarize_scalar_tensor(pre_update_entropy),
        "pre_update_margin_summary": _summarize_scalar_tensor(pre_update_margin),
        "flipped_entropy_summary": _summarize_scalar_tensor(pre_update_entropy[~agreements]),
        "non_flipped_entropy_summary": _summarize_scalar_tensor(pre_update_entropy[agreements]),
        "flipped_margin_summary": _summarize_scalar_tensor(pre_update_margin[~agreements]),
        "non_flipped_margin_summary": _summarize_scalar_tensor(pre_update_margin[agreements]),
        "flipped_entropy_histogram": _summarize_named_histogram(
            pre_update_entropy[~agreements],
            buckets=entropy_histogram_buckets,
        ),
        "non_flipped_entropy_histogram": _summarize_named_histogram(
            pre_update_entropy[agreements],
            buckets=entropy_histogram_buckets,
        ),
        "flipped_margin_histogram": _summarize_named_histogram(
            pre_update_margin[~agreements],
            buckets=margin_histogram_buckets,
        ),
        "non_flipped_margin_histogram": _summarize_named_histogram(
            pre_update_margin[agreements],
            buckets=margin_histogram_buckets,
        ),
        "position_buckets": position_buckets,
        "macro_area_buckets": macro_area_bucket_summary,
    }


def overfit_fixed_rollout(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
    fixed_rollout: FixedRolloutBatch,
    grid_shape: tuple[int, int] = (32, 32),
    reward_source: str = "analytic_hpwl",
    max_steps: int = 1,
    learning_rate: float = 1e-3,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    num_updates: int = 100,
    eval_every: int = 10,
    seed: int = 0,
    adapter_kwargs: dict[str, object] | None = None,
) -> dict[str, object]:
    transitions = fixed_rollout.transitions
    returns = compute_discounted_returns([step.reward for step in transitions], gamma=gamma)
    batch = build_policy_input_batch([step.observation for step in transitions], schema=fixed_rollout.schema)
    action_index = torch.tensor([step.action_index for step in transitions], dtype=torch.long)
    old_log_prob = torch.tensor([step.log_prob for step in transitions], dtype=torch.float32)
    old_value = torch.tensor([step.value for step in transitions], dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def evaluate(current_seed: int) -> dict[str, object]:
        return collect_policy_rollout(
            problem=problem,
            policy=policy,
            grid_shape=grid_shape,
            reward_source=reward_source,
            max_steps=max_steps,
            seed=current_seed,
            deterministic=True,
            adapter_kwargs=adapter_kwargs,
        )

    initial_eval = evaluate(seed)
    history: list[dict[str, float]] = []
    for update_index in range(max(1, num_updates)):
        logits, value_train_space = policy.forward_train_space(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        new_log_prob = distribution.log_prob(action_index)
        entropy = distribution.entropy().mean()
        advantages = normalize_advantages(returns - old_value)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_targets_train_space = policy.encode_value_targets(returns)
        value_loss = torch.nn.functional.mse_loss(value_train_space.squeeze(-1), value_targets_train_space)
        total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (update_index + 1) % max(eval_every, 1) == 0 or update_index == 0 or update_index == num_updates - 1:
            rollout_eval = evaluate(seed)
            history.append(
                {
                    "update": float(update_index + 1),
                    "episode_return": float(rollout_eval["episode_return"]),
                    "final_hpwl": float(rollout_eval["final_hpwl"]),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )

    final_eval = evaluate(seed)
    return {
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "history": history,
    }


def compute_discounted_returns(rewards: list[float], *, gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + (gamma * running)
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def normalize_advantages(advantages: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    if advantages.numel() == 0:
        return advantages
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    if not torch.isfinite(std) or float(std.item()) < eps:
        return torch.zeros_like(advantages)
    return (advantages - mean) / (std + eps)


def _value_target_transform(
    returns: torch.Tensor,
    *,
    target_transform: str,
) -> tuple[torch.Tensor, dict[str, float], callable]:
    if target_transform == "raw":
        def inverse(value_prediction: torch.Tensor) -> torch.Tensor:
            return value_prediction

        return returns.clone(), {
            "target_transform": "raw",
            "target_mean": 0.0,
            "target_std": 1.0,
        }, inverse

    if target_transform == "zscore":
        mean = returns.mean() if returns.numel() else torch.tensor(0.0, dtype=returns.dtype)
        std = returns.std(unbiased=False) if returns.numel() else torch.tensor(1.0, dtype=returns.dtype)
        safe_std = std if torch.isfinite(std) and float(std.item()) > 1e-8 else torch.tensor(1.0, dtype=returns.dtype)

        def inverse(value_prediction: torch.Tensor) -> torch.Tensor:
            return (value_prediction * safe_std) + mean

        return (returns - mean) / safe_std, {
            "target_transform": "zscore",
            "target_mean": float(mean.item()),
            "target_std": float(safe_std.item()),
        }, inverse

    raise ValueError(f"Unknown value target transform: {target_transform}")


def _value_prefit_prediction_metrics(
    *,
    policy: PPOPlacementPolicy,
    batch: PolicyInputBatch,
    raw_returns: torch.Tensor,
    prediction_inverse,
) -> dict[str, object]:
    with torch.no_grad():
        value = policy.predict_value_raw(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
    raw_prediction = value.squeeze(-1).detach().to(torch.float32)
    return {
        "prediction_summary_raw": _summarize_scalar_tensor(raw_prediction),
        "value_return_correlation": _safe_correlation(raw_prediction, raw_returns),
        "raw_mse": float(torch.nn.functional.mse_loss(raw_prediction, raw_returns).item()),
    }


def summarize_rollout_return_signal(
    *,
    fixed_rollouts: list[FixedRolloutBatch],
    total_macro_count: int,
    gamma: float = 0.99,
) -> dict[str, float]:
    active_rollouts = [rollout for rollout in fixed_rollouts if rollout.transitions]
    if not active_rollouts:
        return {
            "rollout_count": 0.0,
            "transition_count": 0.0,
            "return_to_go_vs_step_index": 0.0,
            "return_to_go_vs_remaining_macros": 0.0,
            "return_to_go_vs_current_hpwl": 0.0,
            "episode_return_vs_final_hpwl": 0.0,
            "episode_return_vs_steps": 0.0,
        }

    step_indices: list[float] = []
    remaining_macros: list[float] = []
    current_hpwls: list[float] = []
    return_to_go: list[float] = []
    episode_returns: list[float] = []
    final_hpwls: list[float] = []
    rollout_steps: list[float] = []

    for rollout in active_rollouts:
        rollout_returns = compute_discounted_returns([step.reward for step in rollout.transitions], gamma=gamma)
        for transition_index, (transition, discounted_return) in enumerate(zip(rollout.transitions, rollout_returns)):
            observation = transition.observation
            step_index = float(observation.get("step_index", transition_index))
            step_indices.append(step_index)
            remaining_macros.append(float(max(int(total_macro_count) - int(step_index), 0)))
            current_hpwls.append(float(observation.get("hpwl", rollout.final_hpwl)))
            return_to_go.append(float(discounted_return.item()))
        episode_returns.append(float(rollout.episode_return))
        final_hpwls.append(float(rollout.final_hpwl))
        rollout_steps.append(float(len(rollout.transitions)))

    return_tensor = torch.tensor(return_to_go, dtype=torch.float32)
    return {
        "rollout_count": float(len(active_rollouts)),
        "transition_count": float(return_tensor.numel()),
        "return_to_go_vs_step_index": _safe_correlation(torch.tensor(step_indices, dtype=torch.float32), return_tensor),
        "return_to_go_vs_remaining_macros": _safe_correlation(torch.tensor(remaining_macros, dtype=torch.float32), return_tensor),
        "return_to_go_vs_current_hpwl": _safe_correlation(torch.tensor(current_hpwls, dtype=torch.float32), return_tensor),
        "episode_return_vs_final_hpwl": _safe_correlation(
            torch.tensor(episode_returns, dtype=torch.float32),
            torch.tensor(final_hpwls, dtype=torch.float32),
        ),
        "episode_return_vs_steps": _safe_correlation(
            torch.tensor(episode_returns, dtype=torch.float32),
            torch.tensor(rollout_steps, dtype=torch.float32),
        ),
    }


def prefit_value_head_from_rollout(
    *,
    policy: PPOPlacementPolicy,
    fixed_rollout: FixedRolloutBatch,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    gamma: float = 0.99,
    lr_warmup_epochs: int = 0,
    target_transform: str = "raw",
) -> dict[str, float]:
    return prefit_value_head_from_rollouts(
        policy=policy,
        fixed_rollouts=[fixed_rollout],
        learning_rate=learning_rate,
        epochs=epochs,
        gamma=gamma,
        lr_warmup_epochs=lr_warmup_epochs,
        target_transform=target_transform,
    )


def prefit_value_head_from_rollouts(
    *,
    policy: PPOPlacementPolicy,
    fixed_rollouts: list[FixedRolloutBatch],
    learning_rate: float = 1e-3,
    epochs: int = 50,
    gamma: float = 0.99,
    lr_warmup_epochs: int = 0,
    target_transform: str = "raw",
) -> dict[str, float]:
    active_rollouts = [rollout for rollout in fixed_rollouts if rollout.transitions]
    if not active_rollouts:
        return {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "epochs": 0.0,
            "rollout_count": 0.0,
            "transition_count": 0.0,
            "target_transform": target_transform,
            "lr_warmup_epochs": float(max(0, int(lr_warmup_epochs))),
            "raw_return_summary": _summarize_scalar_tensor(torch.empty((0,), dtype=torch.float32)),
            "target_summary": _summarize_scalar_tensor(torch.empty((0,), dtype=torch.float32)),
            "initial_prediction_summary_raw": _summarize_scalar_tensor(torch.empty((0,), dtype=torch.float32)),
            "final_prediction_summary_raw": _summarize_scalar_tensor(torch.empty((0,), dtype=torch.float32)),
            "initial_value_return_correlation": 0.0,
            "final_value_return_correlation": 0.0,
            "raw_return_variance": 0.0,
            "initial_raw_mse": 0.0,
            "final_raw_mse": 0.0,
        }

    observations: list[dict[str, object]] = []
    all_returns: list[torch.Tensor] = []
    for rollout in active_rollouts:
        observations.extend([step.observation for step in rollout.transitions])
        all_returns.append(compute_discounted_returns([step.reward for step in rollout.transitions], gamma=gamma))
    schema = active_rollouts[0].schema
    batch = build_policy_input_batch(observations, schema=schema)
    raw_returns = torch.cat(all_returns, dim=0).to(torch.float32)
    targets, transform_metadata, inverse_prediction = _value_target_transform(
        raw_returns,
        target_transform=target_transform,
    )
    policy.set_value_target_stats(
        mean=float(transform_metadata["target_mean"]),
        std=float(transform_metadata["target_std"]),
    )

    named_params = list(policy.named_parameters())
    original_requires_grad = {name: param.requires_grad for name, param in named_params}
    for name, param in named_params:
        param.requires_grad = name.startswith("value_head.")

    optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=learning_rate)
    scheduler = None
    warmup_epochs = max(0, int(lr_warmup_epochs))
    if warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

    def compute_loss() -> torch.Tensor:
        _, value_train_space = policy.forward_train_space(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
        return torch.nn.functional.mse_loss(value_train_space.squeeze(-1), targets)

    initial_prediction_metrics = _value_prefit_prediction_metrics(
        policy=policy,
        batch=batch,
        raw_returns=raw_returns,
        prediction_inverse=inverse_prediction,
    )
    initial_loss = float(compute_loss().item())
    for _ in range(max(1, epochs)):
        loss = compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    final_loss = float(compute_loss().item())
    final_prediction_metrics = _value_prefit_prediction_metrics(
        policy=policy,
        batch=batch,
        raw_returns=raw_returns,
        prediction_inverse=inverse_prediction,
    )

    for name, param in named_params:
        param.requires_grad = original_requires_grad[name]

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "epochs": float(max(1, epochs)),
        "rollout_count": float(len(active_rollouts)),
        "transition_count": float(sum(len(rollout.transitions) for rollout in active_rollouts)),
        "target_transform": str(transform_metadata["target_transform"]),
        "lr_warmup_epochs": float(warmup_epochs),
        "raw_return_summary": _summarize_scalar_tensor(raw_returns),
        "target_summary": _summarize_scalar_tensor(targets),
        "initial_prediction_summary_raw": initial_prediction_metrics["prediction_summary_raw"],
        "final_prediction_summary_raw": final_prediction_metrics["prediction_summary_raw"],
        "initial_value_return_correlation": float(initial_prediction_metrics["value_return_correlation"]),
        "final_value_return_correlation": float(final_prediction_metrics["value_return_correlation"]),
        "raw_return_variance": float(raw_returns.var(unbiased=False).item()) if raw_returns.numel() else 0.0,
        "initial_raw_mse": float(initial_prediction_metrics["raw_mse"]),
        "final_raw_mse": float(final_prediction_metrics["raw_mse"]),
        "target_mean": float(transform_metadata["target_mean"]),
        "target_std": float(transform_metadata["target_std"]),
    }


def build_policy_input_batch(
    observations: list[dict[str, object]],
    *,
    schema: FeatureSchema,
) -> PolicyInputBatch:
    graphs = [encode_problem_graph(observation["graph"], schema) for observation in observations]
    return PolicyInputBatch(
        graph_batch=batch_graph_tensors(graphs),
        canvas=torch.stack([observation["canvas"].float() for observation in observations], dim=0),
        current_macro_index=torch.tensor(
            [int(observation["current_macro_index"]) for observation in observations],
            dtype=torch.long,
        ),
        action_mask=torch.stack([observation["action_mask"].bool() for observation in observations], dim=0),
    )
