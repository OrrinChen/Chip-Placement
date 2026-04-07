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


def test_quadratic_solver_preserves_fixed_nodes():
    graph = ProblemGraph(
        nodes={
            "fixed": {"x": 1.0, "y": 2.0},
            "movable": {"x": 9.0, "y": 8.0},
        },
        edges=[("fixed", "movable", {"weight": 1.0})],
        global_features={"bounds": (0.0, 10.0, 0.0, 10.0), "fixed_nodes": {"fixed"}},
    )

    placement = QuadraticPlacementSolver().solve(graph)

    assert placement["fixed"] == (1.0, 2.0)


def test_quadratic_solver_uses_iterative_fallback_for_large_graphs():
    graph = ProblemGraph(
        nodes={
            "fixed": {"x": 1.0, "y": 2.0},
            "movable": {"x": 9.0, "y": 8.0},
            "movable_2": {"x": 8.0, "y": 7.0},
        },
        edges=[
            ("fixed", "movable", {"weight": 1.0}),
            ("fixed", "movable_2", {"weight": 1.0}),
            ("movable", "movable_2", {"weight": 1.0}),
        ],
        global_features={"bounds": (0.0, 10.0, 0.0, 10.0), "fixed_nodes": {"fixed"}},
    )

    placement = QuadraticPlacementSolver(max_direct_nodes=1, relaxation_steps=5).solve(graph)

    assert placement["fixed"] == (1.0, 2.0)
    assert placement["movable"] != (9.0, 8.0)
    assert placement["movable_2"] != (8.0, 7.0)
