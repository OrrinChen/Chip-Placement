from engine.constraints import ConstraintRegistry


def test_constraint_registry_aggregates_penalties():
    registry = ConstraintRegistry()
    registry.register("positive", lambda value: [] if value > 0 else [{"magnitude": 2.0}], 3.0)

    report = registry.check_all(-1)

    assert report.is_valid is False
    assert len(report.violations) == 1
    assert report.total_penalty == 6.0
