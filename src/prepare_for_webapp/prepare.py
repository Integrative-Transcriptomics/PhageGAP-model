"""Utilities to prepare pooled embeddings"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from typing import Dict, List, Tuple
import logging
from openTSNE import TSNE


logger = logging.getLogger(__name__)


def perform_pca(embeddings: Dict[str, np.ndarray], type: str, n_comps: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    """Perform PCA on the protein embeddings

    Params
    ----------
        embeddings (Dict[str, np.ndarray]): Dictionary of protein embeddings mapping protein_ID -> embedding np.ndarray
        type (str): PCA method, specified in config.yaml
        n_comps (int): Number of components for PCA, specified in config.yaml
        seed (int): Seed for reproducibility, specified in config.yaml

    Returns
    ----------
        coords (np.ndarray): PCA coordinates / transformed values of shape (n_samples, n_comps)
        y (List[str]): List of corresponding protein_IDs
    """
    assert n_comps > 1, "need at least 2 components"
    assert type in ["normal", "incremental"], "PCA type must be one of ('normal', 'incremental')"

    X, y = [], []
    for key, value in embeddings.items():
        X.append(value)
        y.append(key)

    shapes = [v.shape for v in X] # v.shape (960,) [esmc] vs. (1024,) [prot_t5]
    print("Unique shapes in embeddings:", set(shapes))

    X = np.stack(X) # shape (n_proteins, 960) [esmc] vs. (n_proteins, 1024) [prot_t5]
    print("Final X shape:", X.shape)

    print("Computing PCA...")
    # instantiate PCA
    if type == "incremental":
        pca = IncrementalPCA(n_components=n_comps)
    else:
        pca = PCA(n_components=n_comps, random_state=seed)
    coords = pca.fit_transform(X)
    logger.info(f"PCA Explained variance: {pca.explained_variance_ratio_}")
    explained = np.sum(pca.explained_variance_ratio_[:n_comps]) * 100
    logger.info(f"PCA total explained variance ({n_comps} comps): {explained:.2f} %")

    return pca, coords, y

def perform_tsne(coords: np.ndarray, perpl: int, metric: str, seed: int) -> np.ndarray:
    """Perform t-SNE on the PCA coordinates
    
    Params
    ----------
        coords (np.ndarray): PCA coordinates / transformed values of shape (n_samples, n_comps)
        perpl (int): Perplexity for t-SNE, specified in config.yaml
        metric (str): Metric for t-SNE, specified in config.yaml
        seed (int): Seed for reproducibility, specified in config.yaml

    Returns
    ----------
        X_tsne (np.ndarray): t-SNE coordinates of shape (n_samples, 2)
    """
    reducer = TSNE(perplexity=perpl, metric=metric, random_state=seed)
    X_tsne = reducer.fit(coords)

    return X_tsne
