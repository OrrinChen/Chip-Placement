from dataclasses import dataclass
import importlib

from engine.contracts import (
    ConstraintChecker,
    DecisionEnv,
    Evaluator,
    SearchStrategy,
    SurrogateModel,
    TrainingStrategy,
)


@dataclass
class DemoDecisionEnv:
    def transition(self, state, action):
        return {"state": state, "action": action}


@dataclass
class DemoConstraintChecker:
    def check(self, state):
        return {"valid": True, "violations": []}


@dataclass
class DemoEvaluator:
    def evaluate(self, candidate):
        return {"score": 1.0}


@dataclass
class DemoSurrogate:
    def predict(self, observation):
        return {"score": 0.5}


@dataclass
class DemoSearchStrategy:
    def solve(self, problem, objective):
        return {"status": "ok", "problem": problem}


@dataclass
class DemoTrainingStrategy:
    def fit(self, samples):
        return {"rows": len(samples)}


def test_demo_objects_satisfy_engine_protocols():
    env: DecisionEnv = DemoDecisionEnv()
    checker: ConstraintChecker = DemoConstraintChecker()
    evaluator: Evaluator = DemoEvaluator()
    surrogate: SurrogateModel = DemoSurrogate()
    search: SearchStrategy = DemoSearchStrategy()
    trainer: TrainingStrategy = DemoTrainingStrategy()

    assert env.transition({}, {})["action"] == {}
    assert checker.check({})["valid"] is True
    assert evaluator.evaluate({})["score"] == 1.0
    assert surrogate.predict({})["score"] == 0.5
    assert search.solve({"nodes": 1}, None)["problem"] == {"nodes": 1}
    assert trainer.fit([1, 2])["rows"] == 2


def test_engine_package_imports_without_optional_rl_env_module():
    engine_package = importlib.import_module("engine")

    assert engine_package.ProblemGraph is not None
