from __future__ import annotations

from typing import Dict, Callable, Any, List
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, precision_recall_fscore_support
import optuna
from src.classify.model import CNN, MLP, CNN_MLP, CNN_Flattened
import torch.optim as optim
import os
import pandas as pd
from collections import defaultdict

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Return classification accuracy (0-1) for logits (N,C) vs targets (N)"""
    assert logits.ndim == 2, "Logits should be a 2D tensor."
    assert targets.ndim == 1, "Targets should be a 1D tensor."
    assert logits.shape[0] == targets.shape[0], "Batch sizes must match."
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.numel()

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> Dict[str, float]:
    """One training pass over 'loader'; returns {loss, acc, mcc, f1}."""
    model.train()
    losses = []
    accs = []
    f1s = []
    precisions = []
    recalls = []
    all_preds = []
    all_targets = []
    all_ids = []
    for batch_idx, (batch_x, batch_y, batch_ids, mask, feats) in enumerate(loader, start=1):

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if feats is not None:
            feats = feats.to(device)
        if mask is not None:
            mask = mask.to(device)
            logits = model(batch_x, mask, feats)
        else:
            logits = model(batch_x, feats=feats)

        loss = loss_fn(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True) # clear gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute batch metrics
        batch_loss = loss.item()
        batch_acc = accuracy(logits.detach(), batch_y)
        preds = torch.argmax(logits.detach(), dim=1)
        batch_precision, batch_recall, batch_f1, support = precision_recall_fscore_support(batch_y.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)

        # Record loss, acc, and preds for overall metrics
        losses.append(batch_loss)
        accs.append(batch_acc)
        f1s.append(batch_f1)
        recalls.append(batch_recall)
        precisions.append(batch_precision)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())
        all_ids.extend(batch_ids)

        # Log every log_interval batches
        if batch_idx % log_interval == 0:
            log_fn(
                {
                    "batch": batch_idx,
                    "loss": batch_loss,
                    "acc": batch_acc,
                    "f1": batch_f1,
                    "recall": batch_recall,
                    "precision": batch_precision
                }
            )
    # integrate into evaluation
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0) if all_preds else 0.0
    macro_precision, macro_recall, macro_f1, support = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0) if all_preds else 0.0

    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs)), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1), "macro_recall": float(macro_recall), "weighted_recall": float(weighted_recall), "macro_precision": float(macro_precision), "weighted_precision": float(weighted_precision), "targets": all_targets, "preds": all_preds, "ids": all_ids, "support": support}


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, labels: List[str]
) -> Dict[str, float]:
    """Evaluate (no grad) over 'loader'; returns {loss, acc, mcc, f1, confusion_matrix, confusion_matrix_norm}."""
    model.eval()
    losses = []
    accs = []
    all_probs = []
    all_preds = []
    all_targets = []
    all_ids = []
    for batch_x, batch_y, batch_ids, mask, feats in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if feats is not None:
            feats = feats.to(device)

        if mask is not None:
            mask = mask.to(device)
            logits = model(batch_x, mask, feats)
        else:
            logits = model(batch_x, feats=feats)
        loss = loss_fn(logits, batch_y)
        # Record loss and accuracy
        losses.append(loss.item())
        accs.append(accuracy(logits, batch_y))
        # Record predictions and targets for Precision, Recall, and F1 score
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=-1)
        # For each row, get sorted descending indices
        #topk = torch.argsort(probs, dim=1, descending=True)
        #top1_idx = topk[:, 0]
        #top1_prob = probs[torch.arange(len(probs), device=probs.device), top1_idx]

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())
        all_ids.extend(batch_ids)

    # integrate into evaluation
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0) if all_preds else 0.0
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0) if all_preds else 0.0
    indiv_precision, indiv_recall, indiv_f1, support = precision_recall_fscore_support(all_targets, all_preds, zero_division=0) if all_preds else 0.0
    multi_confusion = confusion_matrix(all_targets, all_preds, labels=labels)
    multi_confusion_norm = confusion_matrix(all_targets, all_preds, labels=labels, normalize="true")
    return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs)), 
    "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1), 
    "macro_recall": float(macro_recall), "weighted_recall": float(weighted_recall), 
    "macro_precision": float(macro_precision), "weighted_precision": float(weighted_precision), 
    "confusion_matrix": multi_confusion, "confusion_matrix_norm": multi_confusion_norm, 
    "targets": all_targets, "preds": all_preds, "ids": all_ids, "probs": all_probs, 
    "indiv_precision": indiv_precision, "indiv_recall": indiv_recall, 
    "indiv_f1": indiv_f1, "support": support}


def save_fct(path: str, epoch: int, encoder: LabelEncoder, train_targets: List[int], train_preds: List[int], train_ids: List[str], 
             val_targets: List[int], val_preds: List[int], val_ids: List[str]):
    """Saves training and validation data, targets and predictions for an epoch"""
    epoch_path = os.path.join(path, f"epoch_{epoch}")
    os.makedirs(epoch_path, exist_ok=True)

    df_train = pd.DataFrame({
        "protein_ID": train_ids,
        "targets": encoder.inverse_transform(train_targets),
        "preds": encoder.inverse_transform(train_preds)
    })

    df_val = pd.DataFrame({
        "protein_ID": val_ids,
        "targets": encoder.inverse_transform(val_targets),
        "preds": encoder.inverse_transform(val_preds)
    })

    df_train.to_csv(os.path.join(epoch_path, "train.tsv"), sep="\t", index=False)
    df_val.to_csv(os.path.join(epoch_path, "val.tsv"), sep="\t", index=False)

def save_train(path: str, epoch: int, encoder: LabelEncoder, train_targets: List[int], train_preds: List[int], train_ids: List[str]):
    """Saves training data, targets and predictions for an epoch"""
    epoch_path = os.path.join(path, f"epoch_{epoch}")
    os.makedirs(epoch_path, exist_ok=True)

    df_train = pd.DataFrame({
        "protein_ID": train_ids,
        "targets": encoder.inverse_transform(train_targets),
        "preds": encoder.inverse_transform(train_preds)
    })

    df_train.to_csv(os.path.join(epoch_path, "train.tsv"), sep="\t", index=False)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    log_interval: int,
    encoder: LabelEncoder,
    path: str
) -> Dict[str, float]:
    """Train for 'epochs', track best val f1 (restoring its weights), then test.
    Returns dict(best_val_f1, test_loss, test_acc, test_mcc, test_f1)."""
    # track best validation F1
    best_val_f1 = -1.0
    best_state = None

    # track state where validation loss was lowest for the first time
    earliest_lowest_val_loss = 100
    early_stopping_state = None
    got_worse_in_between = False

    epoch30_state = None

    log_objects = defaultdict(dict)
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, log_fn, log_interval
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, encoder.transform(encoder.classes_))

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_weighted_recall": train_metrics["weighted_recall"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_weighted_precision": train_metrics["weighted_precision"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_weighted_recall": val_metrics["weighted_recall"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_weighted_precision": val_metrics["weighted_precision"],
            "time_sec": elapsed_time,
        }
        log_objects[epoch] = log_obj
        log_fn(log_obj)
        save_fct(path, epoch, encoder, train_metrics["targets"], train_metrics["preds"], train_metrics["ids"], val_metrics["targets"], val_metrics["preds"], val_metrics["ids"])

        # best validation F1
        if val_metrics["weighted_f1"] > best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            # Use a deepcopy to ensure the state is fully independent, save model state_dict
            best_state = copy.deepcopy(model.state_dict())

        if val_metrics["loss"] > earliest_lowest_val_loss:
            got_worse_in_between = True

        # early stopping
        if (val_metrics["loss"] < earliest_lowest_val_loss) and (got_worse_in_between != True):
            earliest_lowest_val_loss = val_metrics["loss"]
            # save model state_dict
            early_stopping_state = copy.deepcopy(model.state_dict())

        # epoch 30
        if epoch == epochs:
            # save model state_dict
            epoch30_state = copy.deepcopy(model.state_dict())


    # Best validation F1 evaluation
    assert best_state is not None, "Training loop failed to produce a best model state."
    model.load_state_dict(best_state)

    test_metrics_bestvalf1 = evaluate(model, test_loader, loss_fn, device, encoder.transform(encoder.classes_))

    # Early stopping evaluation
    assert early_stopping_state is not None, "Training loop failed to produce an early stopping state."
    model.load_state_dict(early_stopping_state)

    test_metrics_earlystopping = evaluate(model, test_loader, loss_fn, device, encoder.transform(encoder.classes_))
    
    # Epoch 30 evaluation
    assert epoch30_state is not None, "Training loop failed to produce an epoch 30 state."
    model.load_state_dict(epoch30_state)

    test_metrics_epoch30 = evaluate(model, test_loader, loss_fn, device, encoder.transform(encoder.classes_))
    

    return {
        "best_val_f1_weighted": best_val_f1,
        "bv_test_loss": test_metrics_bestvalf1["loss"],
        "bv_test_acc": test_metrics_bestvalf1["acc"],
        "bv_test_macro_f1": test_metrics_bestvalf1["macro_f1"],
        "bv_test_weighted_f1": test_metrics_bestvalf1["weighted_f1"],
        "bv_test_macro_recall": test_metrics_bestvalf1["macro_recall"],
        "bv_test_weighted_recall": test_metrics_bestvalf1["weighted_recall"],
        "bv_test_macro_precision": test_metrics_bestvalf1["macro_precision"],
        "bv_test_weighted_precision": test_metrics_bestvalf1["weighted_precision"],
        "bv_test_confusion_matrix": test_metrics_bestvalf1["confusion_matrix"],
        "bv_test_confusion_matrix_norm": test_metrics_bestvalf1["confusion_matrix_norm"],
        "bv_test_targets": encoder.inverse_transform(test_metrics_bestvalf1["targets"]),
        "bv_test_preds": encoder.inverse_transform(test_metrics_bestvalf1["preds"]),
        "bv_test_probs": test_metrics_bestvalf1["probs"],
        "bv_test_ids": test_metrics_bestvalf1["ids"],
        "bv_test_indiv_f1": test_metrics_bestvalf1["indiv_f1"],
        "bv_test_indiv_recall": test_metrics_bestvalf1["indiv_recall"],
        "bv_test_indiv_precision": test_metrics_bestvalf1["indiv_precision"],
        "bv_test_support": test_metrics_bestvalf1["support"],
        "bv_best_state": best_state,

        "earliest_lowest_val_loss": earliest_lowest_val_loss,
        "es_test_loss": test_metrics_earlystopping["loss"],
        "es_test_acc": test_metrics_earlystopping["acc"],
        "es_test_macro_f1": test_metrics_earlystopping["macro_f1"],
        "es_test_weighted_f1": test_metrics_earlystopping["weighted_f1"],
        "es_test_macro_recall": test_metrics_earlystopping["macro_recall"],
        "es_test_weighted_recall": test_metrics_earlystopping["weighted_recall"],
        "es_test_macro_precision": test_metrics_earlystopping["macro_precision"],
        "es_test_weighted_precision": test_metrics_earlystopping["weighted_precision"],
        "es_test_confusion_matrix": test_metrics_earlystopping["confusion_matrix"],
        "es_test_confusion_matrix_norm": test_metrics_earlystopping["confusion_matrix_norm"],
        "es_test_targets": encoder.inverse_transform(test_metrics_earlystopping["targets"]),
        "es_test_preds": encoder.inverse_transform(test_metrics_earlystopping["preds"]),
        "es_test_ids": test_metrics_earlystopping["ids"],
        "es_test_probs": test_metrics_earlystopping["probs"],
        "es_test_indiv_f1": test_metrics_earlystopping["indiv_f1"],
        "es_test_indiv_recall": test_metrics_earlystopping["indiv_recall"],
        "es_test_indiv_precision": test_metrics_earlystopping["indiv_precision"],
        "es_test_support": test_metrics_earlystopping["support"],
        "es_state": early_stopping_state,

        "e30_test_loss": test_metrics_epoch30["loss"],
        "e30_test_acc": test_metrics_epoch30["acc"],
        "e30_test_macro_f1": test_metrics_epoch30["macro_f1"],
        "e30_test_weighted_f1": test_metrics_epoch30["weighted_f1"],
        "e30_test_macro_recall": test_metrics_epoch30["macro_recall"],
        "e30_test_weighted_recall": test_metrics_epoch30["weighted_recall"],
        "e30_test_macro_precision": test_metrics_epoch30["macro_precision"],
        "e30_test_weighted_precision": test_metrics_epoch30["weighted_precision"],
        "e30_test_confusion_matrix": test_metrics_epoch30["confusion_matrix"],
        "e30_test_confusion_matrix_norm": test_metrics_epoch30["confusion_matrix_norm"],
        "e30_test_targets": encoder.inverse_transform(test_metrics_epoch30["targets"]),
        "e30_test_preds": encoder.inverse_transform(test_metrics_epoch30["preds"]),
        "e30_test_ids": test_metrics_epoch30["ids"],
        "e30_test_probs": test_metrics_epoch30["probs"],
        "e30_test_indiv_f1": test_metrics_epoch30["indiv_f1"],
        "e30_test_indiv_recall": test_metrics_epoch30["indiv_recall"],
        "e30_test_indiv_precision": test_metrics_epoch30["indiv_precision"],
        "e30_test_support": test_metrics_epoch30["support"],
        "e30_state": early_stopping_state,

        "log_objects": log_objects
    }


def fit_cv(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    log_interval: int,
    encoder: LabelEncoder,
    path: str
) -> Dict[str, float]:
    """Train for 'epochs', track best val f1 (restoring its weights), then test.
    Returns dict(best_val_f1, test_loss, test_acc, test_mcc, test_f1)."""
    # track best validation F1
    best_val_f1 = -1.0
    best_state = None

    log_objects = defaultdict(dict)
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, log_fn, log_interval
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, encoder.transform(encoder.classes_))

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_weighted_recall": train_metrics["weighted_recall"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_weighted_precision": train_metrics["weighted_precision"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_weighted_recall": val_metrics["weighted_recall"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_weighted_precision": val_metrics["weighted_precision"],
            "time_sec": elapsed_time,
        }
        log_objects[epoch] = log_obj
        log_fn(log_obj)
        save_fct(path, epoch, encoder, train_metrics["targets"], train_metrics["preds"], train_metrics["ids"], val_metrics["targets"], val_metrics["preds"], val_metrics["ids"])

        # best validation F1
        if val_metrics["weighted_f1"] > best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            best_val_f1_epoch = epoch

    return log_objects, best_val_f1_epoch, best_val_f1

def fit_test(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    log_interval: int,
    encoder: LabelEncoder,
    path: str
) -> Dict[str, float]:
    """Train for 'epochs', track best val f1 (restoring its weights), then test.
    Returns dict(best_val_f1, test_loss, test_acc, test_mcc, test_f1)."""

    log_objects = defaultdict(dict)
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device, log_fn, log_interval
        )

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_weighted_recall": train_metrics["weighted_recall"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_weighted_precision": train_metrics["weighted_precision"],
            "time_sec": elapsed_time,
        }
        log_objects[epoch] = log_obj
        log_fn(log_obj)
        save_train(path, epoch, encoder, train_metrics["targets"], train_metrics["preds"], train_metrics["ids"])

    model_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_state)

    test_metrics_bestvalf1 = evaluate(model, test_loader, loss_fn, device, encoder.transform(encoder.classes_))

    return {
        "bv_test_loss": test_metrics_bestvalf1["loss"],
        "bv_test_acc": test_metrics_bestvalf1["acc"],
        "bv_test_macro_f1": test_metrics_bestvalf1["macro_f1"],
        "bv_test_weighted_f1": test_metrics_bestvalf1["weighted_f1"],
        "bv_test_macro_recall": test_metrics_bestvalf1["macro_recall"],
        "bv_test_weighted_recall": test_metrics_bestvalf1["weighted_recall"],
        "bv_test_macro_precision": test_metrics_bestvalf1["macro_precision"],
        "bv_test_weighted_precision": test_metrics_bestvalf1["weighted_precision"],
        "bv_test_confusion_matrix": test_metrics_bestvalf1["confusion_matrix"],
        "bv_test_confusion_matrix_norm": test_metrics_bestvalf1["confusion_matrix_norm"],
        "bv_test_targets": encoder.inverse_transform(test_metrics_bestvalf1["targets"]),
        "bv_test_preds": encoder.inverse_transform(test_metrics_bestvalf1["preds"]),
        "bv_test_probs": test_metrics_bestvalf1["probs"],
        "bv_test_ids": test_metrics_bestvalf1["ids"],
        "bv_test_indiv_f1": test_metrics_bestvalf1["indiv_f1"],
        "bv_test_indiv_recall": test_metrics_bestvalf1["indiv_recall"],
        "bv_test_indiv_precision": test_metrics_bestvalf1["indiv_precision"],
        "bv_test_support": test_metrics_bestvalf1["support"],
        "bv_best_state": model_state,

        "log_objects": log_objects
    }


def train_epoch_optuna(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    batch_size_train: int,
    number_train_examples: int,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> None:
    """One training pass over 'loader'"""
    model.train()
    losses = []
    f1s = []
    all_preds = []
    all_targets = []

    for batch_idx, (batch_x, batch_y, ids, mask, feats) in enumerate(train_loader, start=1):

        # stop when number_train_examples are reached
        if batch_idx * batch_size_train > number_train_examples:
            break

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if feats is not None:
            feats = feats.to(device)
        if mask is not None:
            mask = mask.to(device)
            logits = model(batch_x, mask, feats)
        else:
            logits = model(batch_x, feats=feats)

        loss = loss_fn(logits, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True) # clear gradients
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute batch metrics
        batch_loss = loss.item()
        preds = torch.argmax(logits.detach(), dim=1)
        batch_f1 = f1_score(batch_y.cpu().numpy(), preds.cpu().numpy(), average="weighted", zero_division=0)

        # Record loss and preds for overall metrics
        losses.append(batch_loss)
        f1s.append(batch_f1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())

        # Log every log_interval batches
        if batch_idx % log_interval == 0:
            log_fn(
                {
                    "batch": batch_idx,
                    "loss": batch_loss,
                    "f1": batch_f1,
                }
            )

    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) if all_preds else 0.0
    return {"loss": float(np.mean(losses)), "f1": float(weighted_f1)}




@torch.no_grad()
def test_optuna(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    batch_size_test: int, 
    number_test_examples: int,
    log_fn: Callable[[Dict[str, float]], None]
) -> Dict[str, float]:
    """Evaluate (no grad) over 'loader'; returns F1"""
    model.eval()

    all_preds = []
    all_targets = []
    for batch_idx, (batch_x, batch_y, ids, mask, feats) in enumerate(loader):

        if batch_idx * batch_size_test > number_test_examples:
            break 

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        if feats is not None:
            feats = feats.to(device)
        if mask is not None:
            mask = mask.to(device)
            logits = model(batch_x, mask, feats)
        else:
            logits = model(batch_x, feats=feats)

        # Record predictions and targets for F1
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.cpu().numpy().tolist())

    # integrate into evaluation
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0) if all_preds else 0.0
    log_fn({"f1": weighted_f1})

    return float(weighted_f1)


def objective_cnn(
    trial: optuna.trial._trial.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    mean_max: bool,
    n_feats: int,
    use_linear_attention: bool,
    use_nonlinear_attention: bool,
    batch_size_train: int,
    number_train_examples: int,
    batch_size_test: int,
    number_test_examples: int,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> Dict[str, float]:
    """Objective function to be optimized by Optuna. Returns test F1 score. Parameter to be maximized."""
    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 6, step=1)
    num_filters = [int(trial.suggest_int("num_filter_"+str(i), 64, 512, step=64)) for i in range(num_conv_layers)]
    kernel_sizes = [int(trial.suggest_int("kernel_size_"+str(i), 3, 11, step=2)) for i in range(num_conv_layers)]
    dilations = [int(trial.suggest_int("dilation_"+str(i), 1, 10, step=1)) for i in range(num_conv_layers)]
    dropout_conv = trial.suggest_float("dropout_conv", 0.05, 0.4, step=0.01)

    # Generate the model
    use_dilation = True
    model = CNN(trial, in_channels, num_conv_layers, num_filters, kernel_sizes, dropout_conv, num_classes, dilations, use_dilation, mean_max, n_feats, use_linear_attention, use_nonlinear_attention).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=0.0001)

    # Train model
    for epoch in range(1, epochs + 1):
        train_epoch_optuna(model, optimizer, train_loader, device, loss_fn, batch_size_train, number_train_examples, log_fn, log_interval)
        f1 = test_optuna(model, val_loader, device, batch_size_test, number_test_examples, log_fn)

        # Pruning (stops trial early if not promising)
        trial.report(f1, epoch)
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return f1


def objective_cnn_flattened(
    trial: optuna.trial._trial.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    mean_max: bool,
    n_feats: int,
    batch_size_train: int,
    number_train_examples: int,
    batch_size_test: int,
    number_test_examples: int,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> Dict[str, float]:
    """Objective function to be optimized by Optuna. Returns test F1 score. Parameter to be maximized."""
    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 6, step=1)
    num_filters = [int(trial.suggest_int("num_filter_"+str(i), 64, 512, step=64)) for i in range(num_conv_layers)]
    kernel_sizes = [int(trial.suggest_int("kernel_size_"+str(i), 3, 11, step=2)) for i in range(num_conv_layers)]
    dilations = [int(trial.suggest_int("dilation_"+str(i), 1, 10, step=1)) for i in range(num_conv_layers)]
    dropout_conv = trial.suggest_float("dropout_conv", 0.05, 0.4, step=0.01)

    # Generate the model
    use_dilation = True
    model = CNN_Flattened(trial, in_channels, num_conv_layers, num_filters, kernel_sizes, dropout_conv, num_classes, dilations, use_dilation, mean_max, n_feats).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=0.0001)

    # Train model
    for epoch in range(1, epochs + 1):
        train_epoch_optuna(model, optimizer, train_loader, device, loss_fn, batch_size_train, number_train_examples, log_fn, log_interval)
        f1 = test_optuna(model, val_loader, device, batch_size_test, number_test_examples, log_fn)

        # Pruning (stops trial early if not promising)
        trial.report(f1, epoch)
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return f1


def objective_mlp(
    trial: optuna.trial._trial.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    in_dim: int,
    num_classes: int,
    n_feats: int,
    batch_size_train: int,
    number_train_examples: int,
    batch_size_test: int,
    number_test_examples: int,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> Dict[str, float]:
    """Objective function to be optimized by Optuna. Returns test F1 score. Parameter to be maximized."""
    # Define range of values to be tested for the hyperparameters
    num_dimensions = trial.suggest_int("num_dimensions", 2, 6, step=1)
    num_neurons = [int(trial.suggest_int("num_neurons_"+str(i), 64, 512, step=64)) for i in range(num_dimensions)]
    dropout = trial.suggest_float("dropout", 0.05, 0.4, step=0.01)

    # Generate the model
    model = MLP(trial, in_dim, num_dimensions, num_neurons, dropout, num_classes, n_feats).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=0.0001)

    # Train model
    for epoch in range(1, epochs + 1):
        train_epoch_optuna(model, optimizer, train_loader, device, loss_fn, batch_size_train, number_train_examples, log_fn, log_interval)
        f1 = test_optuna(model, val_loader, device, batch_size_test, number_test_examples, log_fn)

        # Pruning (stops trial early if not promising)
        trial.report(f1, epoch)
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return f1

def objective_cnn_mlp(
    trial: optuna.trial._trial.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    in_channels: int,
    num_classes: int,
    n_feats: int,
    use_linear_attention: bool,
    use_nonlinear_attention: bool,
    batch_size_train: int,
    number_train_examples: int,
    batch_size_test: int,
    number_test_examples: int,
    log_fn: Callable[[Dict[str, float]], None],
    log_interval: int
) -> Dict[str, float]:
    """Objective function to be optimized by Optuna. Returns test F1 score. Parameter to be maximized."""
    # Define range of values to be tested for the hyperparameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 6, step=1)
    num_filters = [int(trial.suggest_int("num_filter_"+str(i), 64, 512, step=64)) for i in range(num_conv_layers)]
    kernel_sizes = [int(trial.suggest_int("kernel_size_"+str(i), 3, 11, step=2)) for i in range(num_conv_layers)]
    dilations = [int(trial.suggest_int("dilation_"+str(i), 1, 10, step=1)) for i in range(num_conv_layers)]
    dropout_conv = trial.suggest_float("dropout_conv", 0.05, 0.4, step=0.01)
    num_dimensions = trial.suggest_int("num_dimensions", 2, 6, step=1)
    num_neurons = [int(trial.suggest_int("num_neurons_"+str(i), 64, 512, step=64)) for i in range(num_dimensions)]
    dropout_mlp = trial.suggest_float("dropout", 0.05, 0.4, step=0.01)

    # Generate the model
    use_dilation = True
    model = CNN_MLP(trial, in_channels, num_conv_layers, num_filters, kernel_sizes, dropout_conv, num_classes, dilations, use_dilation, use_linear_attention, use_nonlinear_attention, num_dimensions, num_neurons, dropout_mlp, n_feats).to(device)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=0.0001)

    # Train model
    for epoch in range(1, epochs + 1):
        train_epoch_optuna(model, optimizer, train_loader, device, loss_fn, batch_size_train, number_train_examples, log_fn, log_interval)
        f1 = test_optuna(model, val_loader, device, batch_size_test, number_test_examples, log_fn)

        # Pruning (stops trial early if not promising)
        trial.report(f1, epoch)
        # Handle pruning based on intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return f1

def fit_final_model(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device,
    log_fn: Callable[[Dict[str, Any]], None],
    log_interval: int
) -> Dict[str, float]:
    """Train for 'epochs', log train metrics."""
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch(
            model, loader, optimizer, loss_fn, device, log_fn, log_interval
        )

        elapsed_time = time.time() - start_time

        log_obj = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_recall": train_metrics["weighted_recall"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_weighted_precision": train_metrics["weighted_precision"],
            "train_macro_precision": train_metrics["macro_precision"],
            "time_sec": elapsed_time,
        }
        log_fn(log_obj)

    # Use a deepcopy to ensure the state is fully independent
    final_state = copy.deepcopy(model.state_dict())

    assert final_state is not None, "Training loop failed to produce a final model state."
    model.load_state_dict(final_state)

