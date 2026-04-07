import torch

from engine.gnn_proxy import GraphBatch
from engine.gnn_proxy import GNNProxy
from engine.gnn_proxy import batch_graph_tensors


def test_batch_graph_tensors_offsets_edges_and_tracks_graph_membership():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0]]),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            },
        ]
    )

    assert isinstance(batch, GraphBatch)
    assert tuple(batch.node_features.shape) == (3, 2)
    assert batch.edge_index.tolist() == [[0, 1], [1, 0]]
    assert batch.graph_index.tolist() == [0, 0, 1]
    assert tuple(batch.graph_features.shape) == (2, 0)
    assert batch.num_graphs == 2


def test_batch_graph_tensors_preserves_graph_level_features():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "graph_features": torch.tensor([0.25, 3.0]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0]]),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
                "graph_features": torch.tensor([0.75, 5.0]),
            },
        ]
    )

    assert batch.graph_features.tolist() == [[0.25, 3.0], [0.75, 5.0]]


def test_gnn_proxy_forward_returns_one_prediction_per_graph():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0]]),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            },
        ]
    )

    model = GNNProxy(node_feature_dim=2, hidden_dim=8, output_dim=2, num_layers=2)
    outputs = model(batch)

    assert tuple(outputs.shape) == (2, 2)


def test_gnn_proxy_forward_accepts_graph_level_features():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "graph_features": torch.tensor([0.1, 2.0]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0]]),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
                "graph_features": torch.tensor([0.3, 4.0]),
            },
        ]
    )

    model = GNNProxy(node_feature_dim=2, hidden_dim=8, output_dim=2, num_layers=2, graph_feature_dim=2)
    outputs = model(batch)

    assert tuple(outputs.shape) == (2, 2)


def test_gnn_proxy_forward_supports_sum_pooling():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0]]),
                "edge_index": torch.empty((2, 0), dtype=torch.long),
            },
        ]
    )

    model = GNNProxy(node_feature_dim=2, hidden_dim=8, output_dim=2, num_layers=2, pooling_mode="sum")
    outputs = model(batch)

    assert tuple(outputs.shape) == (2, 2)


def test_gnn_proxy_forward_supports_mean_max_std_pooling():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "graph_features": torch.tensor([0.1, 2.0]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
                "edge_index": torch.tensor([[0], [1]]),
                "graph_features": torch.tensor([0.3, 4.0]),
            },
        ]
    )

    model = GNNProxy(
        node_feature_dim=2,
        hidden_dim=8,
        output_dim=2,
        num_layers=2,
        graph_feature_dim=2,
        pooling_mode="mean-max-std",
    )
    outputs = model(batch)

    assert tuple(outputs.shape) == (2, 2)


def test_gnn_proxy_forward_supports_zeroed_node_path():
    batch = batch_graph_tensors(
        [
            {
                "node_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "edge_index": torch.tensor([[0, 1], [1, 0]]),
                "graph_features": torch.tensor([0.1, 2.0]),
            },
            {
                "node_features": torch.tensor([[2.0, 2.0], [3.0, 3.0]]),
                "edge_index": torch.tensor([[0], [1]]),
                "graph_features": torch.tensor([0.3, 4.0]),
            },
        ]
    )

    model = GNNProxy(
        node_feature_dim=2,
        hidden_dim=8,
        output_dim=2,
        num_layers=2,
        graph_feature_dim=2,
        node_path_scale=0.0,
    )
    outputs = model(batch)

    assert tuple(outputs.shape) == (2, 2)


def test_gnn_proxy_can_overfit_tiny_regression_dataset():
    graphs = [
        {
            "node_features": torch.tensor([[0.0], [0.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        },
        {
            "node_features": torch.tensor([[1.0], [1.0]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        },
    ]
    targets = torch.tensor([[0.0], [2.0]])
    model = GNNProxy(node_feature_dim=1, hidden_dim=16, output_dim=1, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(300):
        batch = batch_graph_tensors(graphs)
        predictions = model(batch)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_predictions = model(batch_graph_tensors(graphs)).detach()
    assert torch.allclose(final_predictions, targets, atol=0.15)
