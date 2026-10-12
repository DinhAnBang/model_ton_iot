"""Controlled Train/Validation training and metrics for Phase 3B."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from ..experimental_phase3a.models import EventMLP, StaticHeteroGraphSAGE


@dataclass
class EvaluationRun:
    model_name: str
    model: torch.nn.Module
    results: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    best_epoch: int
    status: str


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _forward(model: torch.nn.Module, data) -> torch.Tensor:
    if isinstance(model, EventMLP):
        return model(data["event"].x)
    x_dict = {node_type: data[node_type].x for node_type in data.node_types}
    edge_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
    return model(x_dict, edge_dict)


def _weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).float()
    weight = torch.ones(2, dtype=torch.float32)
    present = counts > 0
    weight[present] = counts[present].sum() / (2.0 * counts[present])
    return weight


def metric_row(model_name: str, block: str, labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    pr_auc = float(average_precision_score(labels, probabilities)) if int((labels == 1).sum()) else float("nan")
    fpr = float(fp / (fp + tn)) if (fp + tn) else float("nan")
    return {
        "model": model_name,
        "validation_block": block,
        "n_events": int(len(labels)),
        "normal": int((labels == 0).sum()),
        "attack": int((labels == 1).sum()),
        "accuracy": float((predictions == labels).mean()) if len(labels) else float("nan"),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pr_auc": pr_auc,
        "false_positive_rate": fpr,
        "confusion_matrix_tn_fp_fn_tp": f"[[{tn}, {fp}], [{fn}, {tp}]]",
        "predicted_normal": int((predictions == 0).sum()),
        "predicted_attack": int((predictions == 1).sum()),
        "evaluation_status": "DIAGNOSTIC",
    }


def _validation_arrays(model: torch.nn.Module, validation_data: dict[str, Any], target_ids: dict[str, list[str]]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    all_y: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for block, graph in validation_data.items():
            logits = _forward(model, graph["data"])
            mapping = graph["mapping"].set_index("event_id")
            indices = mapping.loc[target_ids[block], "tensor_index"].to_numpy(dtype=np.int64)
            y = graph["data"]["event"].y[indices].cpu().numpy().astype(np.int64, copy=True)
            prob = torch.softmax(logits[indices], dim=1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(np.int64)
            rows.append(metric_row(model.__class__.__name__, block, y, prob, pred))
            all_y.append(y)
            all_prob.append(prob)
            all_pred.append(pred)
    return rows, np.concatenate(all_y), np.concatenate(all_prob), np.concatenate(all_pred)


def _model_input(model_name: str, train_data):
    if model_name == "EventMLP":
        return EventMLP(int(train_data["event"].x.shape[1]))
    return StaticHeteroGraphSAGE(train_data.metadata())


def train_and_validate(
    model_name: str,
    train_data,
    validation_data: dict[str, Any],
    target_ids: dict[str, list[str]],
    *,
    seed: int = 42,
    epochs: int = 12,
    patience: int = 3,
    learning_rate: float = 0.01,
) -> EvaluationRun:
    set_seed(seed)
    model = _model_input(model_name, train_data)
    # Materialize PyG lazy relation dimensions before optimizer creation.
    with torch.no_grad():
        _ = _forward(model, train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(weight=_weights(train_data["event"].y))
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    wait = 0
    diagnostics: list[dict[str, Any]] = []
    run_status = "PASS"
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = _forward(model, train_data)
        loss = criterion(logits, train_data["event"].y)
        loss_finite = bool(torch.isfinite(loss).item())
        if not loss_finite:
            run_status = "FAIL"
            diagnostics.append({"model": model_name, "epoch": epoch, "train_loss": float("nan"), "status": "FAIL_NAN_LOSS"})
            break
        loss.backward()
        gradients = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
        gradient_norm = float(torch.sqrt(sum(torch.sum(value * value) for value in gradients)).item()) if gradients else 0.0
        max_gradient = float(max((value.abs().max().item() for value in gradients), default=0.0))
        gradients_finite = bool(all(torch.isfinite(value).all().item() for value in gradients))
        if not gradients_finite or max_gradient > 1e6:
            run_status = "FAIL"
            diagnostics.append({"model": model_name, "epoch": epoch, "train_loss": float(loss.item()), "gradient_norm": gradient_norm, "max_gradient": max_gradient, "status": "FAIL_GRADIENT"})
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        validation_rows, y, prob, pred = _validation_arrays(model, validation_data, target_ids)
        val_macro = float(f1_score(y, pred, average="macro", zero_division=0))
        val_pr = float(average_precision_score(y, prob)) if int((y == 1).sum()) else float("nan")
        improved = val_macro > best_score
        if improved:
            best_score = val_macro
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        diagnostics.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "validation_macro_f1": val_macro,
            "validation_pr_auc": val_pr,
            "gradient_norm": gradient_norm,
            "max_gradient": max_gradient,
            "loss_finite": loss_finite,
            "gradients_finite": gradients_finite,
            "learning_rate": learning_rate,
            "class_weight_normal": float(criterion.weight[0].item()),
            "class_weight_attack": float(criterion.weight[1].item()),
            "early_stopping_metric": "validation_macro_f1",
            "best_epoch_so_far": best_epoch,
            "selected_best": improved,
            "status": "PASS",
        })
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    rows, _, _, _ = _validation_arrays(model, validation_data, target_ids)
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for block, graph in validation_data.items():
            logits = _forward(model, graph["data"])
            mapping = graph["mapping"].set_index("event_id")
            indices = mapping.loc[target_ids[block], "tensor_index"].to_numpy(dtype=np.int64)
            all_labels.append(graph["data"]["event"].y[indices].cpu().numpy().astype(np.int64, copy=True))
            probs = torch.softmax(logits[indices], dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_predictions.append((probs >= 0.5).astype(np.int64))
    rows.append(metric_row(model_name, "ALL_VALIDATION_BLOCKS", np.concatenate(all_labels), np.concatenate(all_probs), np.concatenate(all_predictions)))
    for row in rows:
        row["best_epoch"] = best_epoch
        row["learning_rate"] = learning_rate
        row["seed"] = seed
        row["epochs_configured"] = epochs
        row["patience"] = patience
        row["test_used"] = False
    return EvaluationRun(model_name, model, rows, diagnostics, best_epoch, run_status)


def majority_baseline(train_data, validation_data: dict[str, Any], target_ids: dict[str, list[str]]) -> list[dict[str, Any]]:
    normal = int((train_data["event"].y == 0).sum())
    attack = int((train_data["event"].y == 1).sum())
    majority = 1 if attack > normal else 0
    rows: list[dict[str, Any]] = []
    all_y: list[np.ndarray] = []
    for block, graph in validation_data.items():
        mapping = graph["mapping"].set_index("event_id")
        indices = mapping.loc[target_ids[block], "tensor_index"].to_numpy(dtype=np.int64)
        y = graph["data"]["event"].y[indices].cpu().numpy().astype(np.int64)
        predictions = np.full(len(y), majority, dtype=np.int64)
        probabilities = predictions.astype(np.float32)
        row = metric_row("MajorityBaseline", block, y, probabilities, predictions)
        row.update({"best_epoch": 0, "learning_rate": 0.0, "seed": 42, "epochs_configured": 0, "patience": 0, "test_used": False, "train_majority_class": majority})
        rows.append(row)
        all_y.append(y)
    # Recompute ALL using the same prediction rule.
    labels = np.concatenate(all_y)
    predictions = np.full(len(labels), majority, dtype=np.int64)
    rows.append(metric_row("MajorityBaseline", "ALL_VALIDATION_BLOCKS", labels, predictions.astype(np.float32), predictions))
    rows[-1].update({"best_epoch": 0, "learning_rate": 0.0, "seed": 42, "epochs_configured": 0, "patience": 0, "test_used": False, "train_majority_class": majority})
    return rows

