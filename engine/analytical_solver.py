from __future__ import annotations

import cvxpy as cp

from engine.graph import ProblemGraph


class QuadraticPlacementSolver:
    def solve(self, graph: ProblemGraph) -> dict[str, tuple[float, float]]:
        node_ids = list(graph.node_ids)
        if not node_ids:
            return {}

        min_x, max_x, min_y, max_y = graph.global_features.get("bounds", (0.0, 1.0, 0.0, 1.0))
        fixed_nodes = set(graph.global_features.get("fixed_nodes", set()))
        x = cp.Variable(len(node_ids))
        y = cp.Variable(len(node_ids))
        index_by_node = {node_id: idx for idx, node_id in enumerate(node_ids)}

        objective_terms = []
        for source, target, features in graph.edges:
            weight = float(features.get("weight", 1.0))
            src_idx = index_by_node[source]
            dst_idx = index_by_node[target]
            objective_terms.append(weight * cp.square(x[src_idx] - x[dst_idx]))
            objective_terms.append(weight * cp.square(y[src_idx] - y[dst_idx]))

        if not objective_terms:
            objective_terms.append(cp.sum_squares(x - (min_x + max_x) / 2.0))
            objective_terms.append(cp.sum_squares(y - (min_y + max_y) / 2.0))

        constraints = [x >= min_x, x <= max_x, y >= min_y, y <= max_y]
        for node_id in fixed_nodes:
            node_features = graph.nodes[node_id]
            constraints.append(x[index_by_node[node_id]] == float(node_features.get("x", min_x)))
            constraints.append(y[index_by_node[node_id]] == float(node_features.get("y", min_y)))
        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
        problem.solve()

        return {node_id: (float(x.value[idx]), float(y.value[idx])) for node_id, idx in index_by_node.items()}
