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
) -> dict[str, object]:
    torch.manual_seed(seed)
    adapter = ChipRLAdapter(problem=problem, grid_shape=grid_shape)
    env = PlacementEnv(adapter=adapter, max_steps=max_steps, reward_source=reward_source)
    observation, _ = env.reset(seed=seed)
    schema = fit_feature_schema([observation["graph"]])
    steps: list[RolloutStep] = []
    invalid_action_count = 0

    while True:
        batch = build_policy_input_batch([observation], schema=schema)
        with torch.no_grad():
            logits, value = policy(
                graph_batch=batch.graph_batch,
                canvas=batch.canvas,
                current_macro_index=batch.current_macro_index,
                action_mask=batch.action_mask,
            )
        action_distribution = torch.distributions.Categorical(logits=logits)
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
            break

    return {
        "steps": len(steps),
        "invalid_action_count": invalid_action_count,
        "transitions": steps,
        "schema": schema,
        "final_hpwl": adapter.current_hpwl(env.state),
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
) -> dict[str, float]:
    rollout = collect_policy_rollout(
        problem=problem,
        policy=policy,
        grid_shape=grid_shape,
        reward_source=reward_source,
        max_steps=max_steps,
        seed=seed,
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
    advantages = returns - old_value
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    value_loss = torch.nn.functional.mse_loss(value.squeeze(-1), returns)
    total_loss = policy_loss + (value_coef * value_loss) - (entropy_coef * entropy)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "steps": float(len(transitions)),
    }


def compute_discounted_returns(rewards: list[float], *, gamma: float) -> torch.Tensor:
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + (gamma * running)
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


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
