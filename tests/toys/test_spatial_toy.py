from toys.spatial.problem import build_spatial_problem


def test_spatial_toy_builds_problem_graph_without_engine_changes():
    graph = build_spatial_problem(
        rectangles={
            "a": {"width": 1, "height": 1, "x": 0, "y": 0},
            "b": {"width": 1, "height": 1, "x": 2, "y": 0},
        }
    )

    assert graph.node_ids == ("a", "b")
    assert graph.global_features["objective"] == "pairwise_distance"
