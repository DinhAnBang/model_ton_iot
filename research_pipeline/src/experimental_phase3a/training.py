"""Forward/backward smoke checks for MLP and static heterogeneous GraphSAGE."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import EventMLP, StaticHeteroGraphSAGE


def _seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def _class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(y, minlength=2).float()
    weights = torch.ones(2, dtype=torch.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (2.0 * counts[present])
    return weights


def _smoke_row(model_name: str, status: str, reason: str = "", **values: Any) -> dict[str, Any]:
    base = {"model": model_name, "status": status, "reason": reason, "claim_scope": "technical smoke only; no Validation/Test claim"}
    base.update(values)
    return base


def _run_model(model: nn.Module, forward_fn, labels: torch.Tensor, model_name: str, steps: int = 8) -> dict[str, Any]:
    # PyG relation layers use lazy input dimensions. Materialize them before
    # taking the parameter snapshot used by the update test.
    with torch.no_grad():
        _ = forward_fn()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(labels))
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    losses: list[float] = []
    grad_norm = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = forward_fn()
        loss = criterion(logits, labels)
        loss.backward()
        grad_norm = float(sum(parameter.grad.detach().norm().item() for parameter in model.parameters() if parameter.grad is not None))
        optimizer.step()
        losses.append(float(loss.detach().item()))
    changed = any(not torch.equal(initial, current.detach()) for initial, current in zip(before, model.parameters()))
    return _smoke_row(
        model_name,
        "TECHNICAL_PASS" if grad_norm > 0 and changed else "FAIL",
        "",
        steps=steps,
        loss_before=losses[0],
        loss_after=losses[-1],
        loss_decreased=losses[-1] < losses[0],
        gradient_norm=grad_norm,
        parameters_changed=changed,
        event_count=int(labels.shape[0]),
        validation_used=False,
        test_used=False,
    )


def run_static_smoke(data, steps: int = 8) -> list[dict[str, Any]]:
    """Run same-event smoke training; this is deliberately not an evaluation."""
    _seed()
    labels = data["event"].y
    rows: list[dict[str, Any]] = []
    mlp = EventMLP(int(data["event"].x.shape[1]))
    rows.append(_run_model(mlp, lambda: mlp(data["event"].x), labels, "EventMLP", steps))
    try:
        graph_model = StaticHeteroGraphSAGE(data.metadata())
        x_dict = {node_type: data[node_type].x for node_type in data.node_types}
        edge_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
        rows.append(_run_model(graph_model, lambda: graph_model(x_dict, edge_dict), labels, "StaticHeteroGraphSAGE", steps))
    except Exception as exc:
        rows.append(_smoke_row("StaticHeteroGraphSAGE", "UNVERIFIED", str(exc), steps=0, event_count=int(labels.shape[0]), validation_used=False, test_used=False))
    return rows

