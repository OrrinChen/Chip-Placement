from dataclasses import dataclass

from engine.contracts import ConstraintChecker, Evaluator, SurrogateModel, TransitionModel


@dataclass
class DemoTransition:
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


def test_demo_objects_satisfy_engine_protocols():
    transition: TransitionModel = DemoTransition()
    checker: ConstraintChecker = DemoConstraintChecker()
    evaluator: Evaluator = DemoEvaluator()
    surrogate: SurrogateModel = DemoSurrogate()

    assert transition.transition({}, {})["action"] == {}
    assert checker.check({})["valid"] is True
    assert evaluator.evaluate({})["score"] == 1.0
    assert surrogate.predict({})["score"] == 0.5
