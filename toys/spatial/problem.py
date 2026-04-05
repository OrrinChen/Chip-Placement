from engine.graph import ProblemGraph


def build_spatial_problem(rectangles: dict[str, dict]) -> ProblemGraph:
    return ProblemGraph(
        nodes=rectangles,
        edges=[],
        global_features={"objective": "pairwise_distance", "mode": "offline"},
    )
