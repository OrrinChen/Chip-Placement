from engine.analytical_solver import QuadraticPlacementSolver
from engine.graph import ProblemGraph


def test_quadratic_solver_returns_in_bounds_coordinates():
    graph = ProblemGraph(
        nodes={"a": {}, "b": {}},
        edges=[("a", "b", {"weight": 1.0})],
        global_features={"bounds": (0.0, 10.0, 0.0, 10.0)},
    )

    solver = QuadraticPlacementSolver()
    placement = solver.solve(graph)

    assert set(placement) == {"a", "b"}
    for x, y in placement.values():
        assert 0.0 <= x <= 10.0
        assert 0.0 <= y <= 10.0
