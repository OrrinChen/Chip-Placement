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


def run_single_ppo_update(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
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
) -> dict[str, float]:
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

    returns = compute_discounted_returns([step.reward for step in transitions], gamma=gamma)
    batch = build_policy_input_batch([step.observation for step in transitions], schema=schema)
    action_index = torch.tensor([step.action_index for step in transitions], dtype=torch.long)
    old_log_prob = torch.tensor([step.log_prob for step in transitions], dtype=torch.float32)
    old_value = torch.tensor([step.value for step in transitions], dtype=torch.float32)

    logits, value = policy(
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
    value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), returns)
    total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)
    approx_kl = (old_log_prob - new_log_prob).mean()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "steps": float(len(transitions)),
        "final_hpwl": float(rollout["final_hpwl"]),
        "invalid_action_count": float(rollout["invalid_action_count"]),
        "episode_return": float(rollout["episode_return"]),
        "completed_episode": 1.0 if rollout["completed_episode"] else 0.0,
        "is_legal": 1.0 if rollout["is_legal"] else 0.0,
        "approx_kl": float(approx_kl.item()),
    }


def train_ppo_policy(
    *,
    problem: ChipProblem,
    policy: PPOPlacementPolicy,
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
) -> dict[str, object]:
    history: list[dict[str, float]] = []
    best_hpwl_so_far: float | None = None
    for update_index in range(max(1, num_updates)):
        update_metrics = run_single_ppo_update(
            problem=problem,
            policy=policy,
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
        logits, value = policy(
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
        value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), returns)
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


def prefit_value_head_from_rollout(
    *,
    policy: PPOPlacementPolicy,
    fixed_rollout: FixedRolloutBatch,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    gamma: float = 0.99,
) -> dict[str, float]:
    transitions = fixed_rollout.transitions
    if not transitions:
        return {"initial_loss": 0.0, "final_loss": 0.0, "epochs": 0.0}

    returns = compute_discounted_returns([step.reward for step in transitions], gamma=gamma)
    batch = build_policy_input_batch([step.observation for step in transitions], schema=fixed_rollout.schema)

    named_params = list(policy.named_parameters())
    original_requires_grad = {name: param.requires_grad for name, param in named_params}
    for name, param in named_params:
        param.requires_grad = name.startswith("value_head.")

    optimizer = torch.optim.Adam(policy.value_head.parameters(), lr=learning_rate)

    def compute_loss() -> torch.Tensor:
        _, value = policy(
            graph_batch=batch.graph_batch,
            canvas=batch.canvas,
            current_macro_index=batch.current_macro_index,
            action_mask=batch.action_mask,
        )
        return torch.nn.functional.mse_loss(value.squeeze(-1), returns)

    initial_loss = float(compute_loss().item())
    for _ in range(max(1, epochs)):
        loss = compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = float(compute_loss().item())

    for name, param in named_params:
        param.requires_grad = original_requires_grad[name]

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "epochs": float(max(1, epochs)),
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
