"""Prediction utilities"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
import torch
import logging
import pandas as pd
from src.classify.model import MLP, CNN, CNN_MLP
from torch.nn.utils.rnn import pad_sequence
import psutil
import time
import os

logger = logging.getLogger(__name__)


def load_model_mlp(path: str, num: int):
    """
    Loads the saved weights, sets the model to evaluation mode, and returns model metadata.

    Returns
    -------
    model (torch.nn.Module): The loaded PyTorch model with weights restored.
    label_map (dict):       Mapping from predicted class indices to human-readable labels. If the checkpoint
                            stores labels as {label: index}, this mapping is inverted to {index: label}.
    cfg (dict):             The configuration dictionary used to initialize the model, taken from the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]  
    label_map = checkpoint["label_map"]

    model = MLP(
        trial=None,
        in_dim=cfg["data"]["embed_dim"],
        num_dimensions=cfg["models"][f"model{num}"]["mlp"]["num_dimensions"],
        num_neurons=cfg["models"][f"model{num}"]["mlp"]["num_neurons"],
        dropout=cfg["models"][f"model{num}"]["mlp"]["dropout"],
        num_classes=len(label_map),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # exchange keys and values for convenience
    label_map = {int(v): str(k) for k, v in label_map.items()}

    return model, label_map, cfg



def load_model_cnn(path: str, num: int, model_type: str):
    """
    Loads the saved weights, sets the model to evaluation mode, and returns model metadata.

    Returns
    -------
    model (torch.nn.Module): The loaded PyTorch model with weights restored.
    label_map (dict):       Mapping from predicted class indices to human-readable labels. If the checkpoint
                            stores labels as {label: index}, this mapping is inverted to {index: label}.
    cfg (dict):             The configuration dictionary used to initialize the model, taken from the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]  
    label_map = checkpoint["label_map"]

    if model_type == "cnn":
        model = CNN(
            trial=None,
            in_channels=cfg["data"]["embed_dim"],
            num_conv_layers=cfg["models"][f"model{num}"]["cnn"]["num_conv_layers"],
            num_filters=cfg["models"][f"model{num}"]["cnn"]["num_filters"],
            kernel_sizes=cfg["models"][f"model{num}"]["cnn"]["kernel_sizes"],
            dropout=cfg["models"][f"model{num}"]["cnn"]["dropout"],
            num_classes=len(label_map),
            dilations=cfg["models"][f"model{num}"]["cnn"]["dilations"],
            use_dilation=cfg["models"][f"model{num}"]["cnn"]["use_dilation"],
            mean_max=cfg["models"][f"model{num}"]["cnn"]["mean_max"],
            n_feats=0,
            use_linear_attention=cfg["models"][f"model{num}"]["cnn"]["use_linear_attention"],
            use_nonlinear_attention=cfg["models"][f"model{num}"]["cnn"]["use_nonlinear_attention"]
        )
    elif model_type == "cnn_mlp":
        model = CNN_MLP(
            trial=None,
            in_channels=cfg["data"]["embed_dim"],
            num_conv_layers=cfg["models"][f"model{num}"]["cnn_mlp"]["num_conv_layers"],
            num_filters=cfg["models"][f"model{num}"]["cnn_mlp"]["num_filters"],
            kernel_sizes=cfg["models"][f"model{num}"]["cnn_mlp"]["kernel_sizes"],
            dropout_conv=cfg["models"][f"model{num}"]["cnn_mlp"]["dropout_conv"],
            num_classes=len(label_map),
            dilations=cfg["models"][f"model{num}"]["cnn_mlp"]["dilations"],
            use_dilation=cfg["models"][f"model{num}"]["cnn_mlp"]["use_dilation"],
            use_linear_attention=cfg["models"][f"model{num}"]["cnn_mlp"]["use_linear_attention"],
            use_nonlinear_attention=cfg["models"][f"model{num}"]["cnn_mlp"]["use_nonlinear_attention"],
            num_dimensions=cfg["models"][f"model{num}"]["cnn_mlp"]["num_dimensions"],
            num_neurons=cfg["models"][f"model{num}"]["cnn_mlp"]["num_neurons"],
            dropout_mlp=cfg["models"][f"model{num}"]["cnn_mlp"]["dropout_mlp"],
        )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # exchange keys and values for convenience
    label_map = {int(v): str(k) for k, v in label_map.items()}

    return model, label_map, cfg


def monitor_model_load(path, num, model_type):
    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / 1024**2
    start = time.perf_counter()

    model, label_map, cfg = load_model_cnn(path, num, model_type)

    end = time.perf_counter()
    mem_after = process.memory_info().rss / 1024**2

    logger.info("\n=== MODEL LOAD STATS ===")
    logger.info(f"Time: {end - start:.2f} s")
    logger.info(f"RAM before: {mem_before:.2f} MB")
    logger.info(f"RAM after:  {mem_after:.2f} MB")
    logger.info(f"Delta RAM:  {mem_after - mem_before:.2f} MB")

    return model, label_map, cfg


def predict(embeddings: Dict[str, np.ndarray | List[np.ndarray]], metadata_df: pd.DataFrame, path: str, num: int, model_type: str) -> pd.DataFrame:
    """
    Predict classes for protein embeddings using the selected model.

    Parameters
    ----------
    embeddings (Dict[str, np.ndarray | List[np.ndarray]]): Dictionary of protein sequence embeddings {protein_ID: embedding}.
    metadata_df (pd.DataFrame): DataFrame with metadata
    path (str): path to model.pt
    num (int): Number of model
    model_type (str): One of "mlp", "cnn", "cnn_mlp"

    Returns
    -----------
    results (pd.DataFrame): DataFrame containing predictions and probabilities.
    """
    assert model_type in ("mlp", "cnn", "cnn_mlp")

    protein_ids = list(embeddings.keys())
    if model_type in ("cnn", "icnn"):
        model, label_map, cfg = monitor_model_load(path, num, model_type)

        # batch predict
        batch_size = 32
        all_probs = []
        all_features = []

        for i in range(0, len(protein_ids), batch_size):
            batch_ids = protein_ids[i:i+batch_size]
            tensor_list = [torch.from_numpy(embeddings[pid]).float() for pid in batch_ids]
            X = pad_sequence(tensor_list, batch_first=True) # shape (batch, max_len, D)
            lengths = torch.tensor([t.shape[0] for t in tensor_list])
            mask = torch.arange(X.size(1))[None, :] < lengths[:, None]

            # Forward pass
            with torch.no_grad():
                logits, features = model(X, mask, feats=None, return_features=True)
                probs = torch.softmax(logits, dim=-1)

            all_features.append(features.cpu())
            all_probs.append(probs)

        probs = torch.cat(all_probs).numpy()
        all_features = torch.cat(all_features)
        filter_dict = {pid: feat.numpy() for pid, feat in zip(protein_ids, all_features)}

    elif model_type == "mlp":
        tensor_idxs = [torch.from_numpy(embeddings[pid]).float() for pid in protein_ids]
        X = torch.stack(tensor_idxs, dim=0) # shape (batch, L)

        model, label_map, _  = load_model_mlp(path, num)

        # Forward pass
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=-1).numpy()

    # For each row, get sorted descending indices
    topk = np.argsort(probs, axis=1)[:, ::-1]

    top1_idx = topk[:, 0]
    top2_idx = topk[:, 1]
    top3_idx = topk[:, 2]

    top1_prob = probs[np.arange(len(probs)), top1_idx]
    top2_prob = probs[np.arange(len(probs)), top2_idx]
    top3_prob = probs[np.arange(len(probs)), top3_idx]

    top1_label = [label_map[int(i)] for i in top1_idx]
    top2_label = [label_map[int(i)] for i in top2_idx]
    top3_label = [label_map[int(i)] for i in top3_idx]

    protein_ids = list(embeddings.keys())

    results = pd.DataFrame({
        "protein_ID": protein_ids,
        "Predicted Class": top1_label,
        "top2": top2_label,
        "top3": top3_label,
        "P(Predicted)": top1_prob,
        "P(top2)": top2_prob,
        "P(top3)": top3_prob
    })

    results = pd.merge(metadata_df, results, on="protein_ID", how="left")

    return results, filter_dict