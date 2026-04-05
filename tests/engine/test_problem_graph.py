from engine.graph import ProblemGraph


def test_problem_graph_preserves_nodes_edges_and_global_features():
    graph = ProblemGraph(
        nodes={"n1": {"kind": "macro", "width": 3}},
        edges=[("n1", "n1", {"weight": 1.0})],
        global_features={"domain": "chip"},
    )

    assert graph.node_ids == ("n1",)
    assert graph.edge_count == 1
    assert graph.global_features["domain"] == "chip"
