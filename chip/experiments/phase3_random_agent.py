from __future__ import annotations

import random

from chip.problem import ChipProblem
from chip.rl_adapter import ChipRLAdapter
from engine.rl_env import PlacementEnv


def run_random_agent_smoke(
    problem: ChipProblem,
    steps: int,
    reward_source: str = "analytic_hpwl",
    seed: int = 0,
) -> dict[str, object]:
    adapter = ChipRLAdapter(problem=problem)
    env = PlacementEnv(adapter=adapter, max_steps=steps, reward_source=reward_source)
    rng = random.Random(seed)
    observation, info = env.reset()
    trace: list[dict[str, object]] = []
    terminated = False
    truncated = False
    while not terminated and not truncated and len(trace) < steps:
        if adapter.current_macro_for_step(len(env.state.placed_macros)) is None:
            break
        action = adapter.sample_random_action(env.state, rng)
        observation, reward, terminated, truncated, step_info = env.step(action)
        trace.append(
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "step_index": step_info["step_index"],
            }
        )
    return {
        "steps": len(trace),
        "trace": trace,
        "final_hpwl": adapter.current_hpwl(env.state),
        "final_observation": observation,
        "reset_info": info,
    }
