from engine.sa_solver import SimulatedAnnealingSolver


def test_sa_solver_improves_simple_scalar_objective():
    solver = SimulatedAnnealingSolver(seed=7)

    result = solver.solve(
        initial_state=10.0,
        objective_fn=lambda value: value * value,
        neighbor_fn=lambda value, rng: value - 1.0,
        iterations=20,
    )

    assert result.best_score <= 1.0
