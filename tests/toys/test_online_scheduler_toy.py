from toys.online_scheduler.problem import build_scheduler_problem


def test_online_scheduler_toy_builds_problem_graph_without_engine_changes():
    graph = build_scheduler_problem(
        jobs={
            "j1": {"arrival": 0, "size": 3, "deadline": 4},
            "j2": {"arrival": 1, "size": 2, "deadline": 5},
        },
        machine_count=2,
    )

    assert graph.node_ids == ("j1", "j2")
    assert graph.global_features["mode"] == "online"
    assert graph.global_features["machine_count"] == 2
