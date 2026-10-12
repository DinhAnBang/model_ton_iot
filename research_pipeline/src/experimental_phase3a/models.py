"""Small static models; no temporal message passing is defined here."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, SAGEConv


class EventMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticHeteroGraphSAGE(nn.Module):
    def __init__(self, metadata: tuple[list[str], list[tuple[str, str, str]]], hidden_dim: int = 32, classes: int = 2) -> None:
        super().__init__()
        _, edge_types = metadata
        self.conv1 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in edge_types}, aggr="sum")
        self.conv2 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in edge_types}, aggr="sum")
        self.classifier = nn.Linear(hidden_dim, classes)

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        hidden = self.conv1(x_dict, edge_index_dict)
        hidden = {key: torch.relu(value) for key, value in hidden.items()}
        hidden = self.conv2(hidden, edge_index_dict)
        return self.classifier(torch.relu(hidden["event"]))

