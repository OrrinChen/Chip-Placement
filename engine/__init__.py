"""Neutral engine package."""

from engine.contracts import (
    ConstraintChecker,
    DecisionEnv,
    Evaluator,
    RLAdapter,
    SearchStrategy,
    SurrogateModel,
    TrainingStrategy,
    TransitionModel,
)
from engine.graph import ProblemGraph
from engine.gnn_proxy import GNNProxy
from engine.gnn_proxy import GraphBatch
from engine.gnn_proxy import batch_graph_tensors
from engine.ppo_policy import PPOPlacementPolicy

try:
    from engine.rl_env import PlacementEnv
except ModuleNotFoundError:  # Phase 2 anchor should not depend on Phase 3 scaffolding files.
    PlacementEnv = None

__all__ = [
    "ConstraintChecker",
    "DecisionEnv",
    "Evaluator",
    "GNNProxy",
    "GraphBatch",
    "ProblemGraph",
    "PPOPlacementPolicy",
    "RLAdapter",
    "SearchStrategy",
    "SurrogateModel",
    "TrainingStrategy",
    "TransitionModel",
    "batch_graph_tensors",
]

if PlacementEnv is not None:
    __all__.append("PlacementEnv")
