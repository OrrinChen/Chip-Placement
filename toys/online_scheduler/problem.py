from engine.graph import ProblemGraph


def build_scheduler_problem(jobs: dict[str, dict], machine_count: int) -> ProblemGraph:
    return ProblemGraph(
        nodes=jobs,
        edges=[],
        global_features={
            "objective": "weighted_tardiness",
            "mode": "online",
            "machine_count": machine_count,
        },
    )
