"""Utility helpers: determinism, HDF5 file saving"""

from __future__ import annotations

import random
import numpy as np
import torch
import logging 
import h5py
from tqdm import tqdm
import pandas as pd
from typing import Dict, Tuple, List, Any, Callable
import os
import json
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import set_seed
import datetime

logger = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
    """Set seeds for python, numpy, transformers and torch.

    Raises
    -------
    AssertionError
        If seed is negative.
    """
    assert seed >= 0, "Seed must be non-negative"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_checkpoint_from_embedding_filename(file_path: str) -> str:
    """Extracts the pLM used for creating protein embeddings from a filename (last two parts)"""
    parts = os.path.splitext(os.path.basename(file_path))[0].split("_")
    checkpoint = "_".join(parts[-2:])

    return checkpoint


def read_hdf5(file_path: str, meta_columns=None) -> Tuple[Dict[str, np.ndarray | List[np.ndarray]], pd.DataFrame]:
    """Read embeddings and selected metadata from an HDF5 file.

    Params:
        file_path (str): Path to HDF5 file
        meta_columns (list, optional): List of metadata columns to extract. If None, extract all.

    Returns:
        embeddings (dict): mapping protein_ID -> np.ndarray (if 1 layer) or List[np.ndarray] (if multiple layers)
        metadata_df (pd.DataFrame): DataFrame with extracted metadata columns
    """
    embeddings = {}
    metadata = []

    logger.info(f"Reading file from {file_path}...")

    with h5py.File(file_path, 'r') as f:
        for pid in f.keys():
            group = f[pid]

            # get embedding 
            emb = np.array(group["embedding"]) # shape (num_layers, L, embed_dim)

            # convert to list if more than 1 layer
            if emb.ndim <= 2:
                # pooled or single-layer (L, D) embeddings 
                embeddings[pid] = emb # shape (L, embed_dim)
                print(f"{pid}: single-layer, shape={emb.shape}")
            elif emb.ndim == 3:
                # multi-layer
                embeddings[pid] = [emb[i] for i in range(emb.shape[0])] # List[(L, embed_dim)]
                print(f"{pid}: multi-layer, num layers={emb.shape[0]}, shape first layer={emb[0].shape}")

            else:
                raise ValueError(f"Unexpected embedding shape {emb.shape}")

            # load metadata
            meta_group = group["metadata"]
            meta_dict = {k: v for k, v in meta_group.attrs.items() if (meta_columns is None) or (k in meta_columns)}
            meta_dict["protein_ID"] = pid
            metadata.append(meta_dict)

    metadata_df = pd.DataFrame(metadata)

    return embeddings, metadata_df


def create_embeddings_file(embs_dict: Dict[str, np.ndarray | List[np.ndarray]], df: pd.DataFrame, output_path: str):
    """Create embeddings file for a DataFrame and store them in HDF5 file with metadata.

    Params:
        embs_dict (Dict[str, np.ndarray | List[np.ndarray]]): Dictionary mapping protein_ID -> embedding
        df (pd.DataFrame): DataFrame containing protein IDs and metadata 
        output_path (str): HDF5 file path to store embeddings
    """
    logger.info(f"Total proteins in df: {len(df)}")
    logger.info(f"Number of embeddings: {len(embs_dict)}")
    missing = set(df["protein_ID"]) - set(embs_dict.keys())
    if missing:
        logger.warning(f"Missing PIDs: {missing}")

    # write to HDF5 file (write mode)
    with h5py.File(output_path, "w") as hdf:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing embeddings"):
            pid = str(row["protein_ID"])
            emb = embs_dict[pid]

            print("Before saving:")
            print(pid, type(emb), emb[0].shape if isinstance(emb, list) else emb.shape)

            # allow overwriting file
            if pid in hdf:
                del hdf[pid]
            
            # create group for each protein (groups work like dicts)
            group = hdf.create_group(pid)

            # store embedding as dataset (datasets work like np.arrays)
            if isinstance(emb, list):
                # multiple layers -> stack into 3D array
                stacked = np.stack(emb)
                group.create_dataset("embedding", data=stacked, compression="gzip")
            else:
                # single layer
                group.create_dataset("embedding", data=emb, compression="gzip")

            # store metadata as attributes
            meta_group = group.create_group("metadata")
            for col in df.columns:
                value = row[col]
                # convert to scalars
                if pd.isna(value):
                    continue
                if hasattr(value, "item"):
                    value = value.item()
                meta_group.attrs[col] = str(value) # store everything as string


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    

def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file with support for NumPy types

    Parameters
    ----------
    data (Dict[str, Any]): The dictionary to save
    path (str): The output file path
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)


def log(logger: logging.Logger) -> Callable[[Dict[str, Any]], None]:
    """Create a logging callable using standard logging"""

    def log_fn(obj: Dict[str, Any]):
        logger.info(
            " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in obj.items()
            )
        )

    return log_fn

def save_indiv_results(results: Dict[str, np.ndarray], path: str) -> None:
    """Save individual evaluation results per class to a CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)

def save_cm_norm_raw(confusion_matrix, labels, path: str) -> None:
    """Save confusion matrix values to a CSV file"""
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    df.to_csv(path, index=False)

def save_cm_norm(confusion_matrix, labels, path: str) -> None:
    """Save a confusion matrix to a PNG"""
    fig, ax = plt.subplots(figsize=(25, 25))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    
    disp.plot(ax=ax, cmap="RdYlGn", values_format=".2f", im_kw={"vmin": 0, "vmax": 1})
    plt.savefig(path)
    plt.close(fig)

def save_cm(confusion_matrix, labels, path: str) -> None:
    """Save a confusion matrix to a PNG"""
    fig, ax = plt.subplots(figsize=(25, 25))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    
    disp.plot(ax=ax, cmap="viridis")
    plt.savefig(path)
    plt.close(fig)

def save_labelmap(label_map: Dict[str, int], path: str) -> None:
    """Save label map to a .txt file"""
    with open(path, "w") as f:
        for label, idx in label_map.items():
            f.write(f"{label}\t{idx}\n")

def save_unknown_IDs(df: pd.DataFrame, path: str) -> None:
    """Save unknown IDs to .csv file"""
    df[["protein_ID"]].to_csv(path, index=False)

def save_optuna_results(study, path: str) -> None:
    """Save optuna study report results to .csv file and return as DataFrame"""
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on F1
    df.to_csv(path, index=False)  # Save to csv file

    return df

def save_test(targets, preds, probs, ids, encoder, path: str):
    """Save test set targets and predictions"""
    labels = encoder.classes_

    df_probs = pd.DataFrame(probs, columns=[f"prob_{label}" for label in labels])
    df_probs["id"] = ids
    df_probs["target"] = targets
    df_probs["pred"] = preds

    """
    df_test = pd.DataFrame({
        "protein_ID": ids,
        "targets": targets,
        "preds": preds,
        "probs": probs
    })
    """
    df_probs.to_csv(path, sep="\t", index=False)
 

def get_output_dir(optimize, context, pooled):
    """get Hydra output directory based on booleans"""
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if optimize:
        return f"outputs/classify_optuna/{now}"
    elif context:
        return f"outputs/classify_genomic_context/{now}"
    elif pooled:
        return f"outputs/classify_pooled/{now}"
    else:
        return f"outputs/classify_unpooled/{now}"
    

    
