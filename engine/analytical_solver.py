from __future__ import annotations

from collections import defaultdict

import cvxpy as cp

from engine.graph import ProblemGraph


class QuadraticPlacementSolver:
    def __init__(self, max_direct_nodes: int = 2000, relaxation_steps: int = 20) -> None:
        self.max_direct_nodes = max_direct_nodes
        self.relaxation_steps = relaxation_steps

    def solve(self, graph: ProblemGraph) -> dict[str, tuple[float, float]]:
        node_ids = list(graph.node_ids)
        if not node_ids:
            return {}

        if len(node_ids) > self.max_direct_nodes:
            return self._solve_iterative(graph)

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

    def _solve_iterative(self, graph: ProblemGraph) -> dict[str, tuple[float, float]]:
        min_x, max_x, min_y, max_y = graph.global_features.get("bounds", (0.0, 1.0, 0.0, 1.0))
        fixed_nodes = set(graph.global_features.get("fixed_nodes", set()))
        positions = {
            node_id: (
                float(graph.nodes[node_id].get("x", (min_x + max_x) / 2.0)),
                float(graph.nodes[node_id].get("y", (min_y + max_y) / 2.0)),
            )
            for node_id in graph.node_ids
        }
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for source, target, features in graph.edges:
            weight = float(features.get("weight", 1.0))
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))

        for _ in range(self.relaxation_steps):
            next_positions = dict(positions)
            for node_id in graph.node_ids:
                if node_id in fixed_nodes:
                    continue
                neighbors = adjacency.get(node_id, [])
                if not neighbors:
                    continue
                total_weight = sum(weight for _, weight in neighbors)
                if total_weight <= 0:
                    continue
                x_value = sum(positions[neighbor][0] * weight for neighbor, weight in neighbors) / total_weight
                y_value = sum(positions[neighbor][1] * weight for neighbor, weight in neighbors) / total_weight
                next_positions[node_id] = (
                    min(max(x_value, min_x), max_x),
                    min(max(y_value, min_y), max_y),
                )
            positions = next_positions
        return positions
