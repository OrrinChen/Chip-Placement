from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProblemGraph:
    nodes: dict[str, dict]
    edges: list[tuple[str, str, dict]] = field(default_factory=list)
    global_features: dict[str, object] = field(default_factory=dict)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self.nodes.keys())

    @property
    def edge_count(self) -> int:
        return len(self.edges)
