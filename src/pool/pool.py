"""Pooling utilities for pLM protein embeddings"""

import logging
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class AttentionPool(nn.Module):
    def __init__(self, emb_dim):
        out_dim = emb_dim
        super().__init__()
        self.attn = nn.Linear(emb_dim, 1)
        self.fc = nn.Linear(emb_dim, out_dim)
    
    def forward(self, x): # x: (L, embed_dim)
        scores = self.attn(x) # (L, 1)
        weights = torch.softmax(scores, 0) # (L, 1)
        pooled = (weights * x).sum(dim=0) # (embed_dim,)
        return self.fc(pooled) # (out_dim,)


def filter_nan_embeddings(embeddings: Dict[str, np.ndarray | List[np.ndarray]], metadata_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Remove embeddings that contain only zeros (as for some pLMs not all proteins could be embedded)"""
    def is_all_nan(embedding):
        if isinstance(embedding, list):
            return all(np.all(np.isnan(e)) for e in embedding)
        else:
            return np.all(np.isnan(embedding))

    # identify zero vector protein IDs
    nan_vector_proteins = [protein_id for protein_id, embedding in embeddings.items() if is_all_nan(embedding)]
    # remove from embed_dict
    filtered_embed_dict = {protein_id: embedding for protein_id, embedding in embeddings.items() if protein_id not in nan_vector_proteins}

    # remove corresponding rows in metadata_df
    filtered_metadata_df = metadata_df[~metadata_df['protein_ID'].isin(nan_vector_proteins)]

    return filtered_embed_dict, filtered_metadata_df


def pool_embeddings(embeddings: Dict[str, List[np.ndarray] | np.ndarray], checkpoint: str, layers: List[int], strategy: str) -> Dict[str, np.ndarray]:
    """Pools specified hidden layers of protein sequence embeddings using a given strategy (e.g. mean/max pooling). 

    Params
    ----------
    embeddings (Dict[str, List[np.ndarray] | np.ndarray]): Dictionary mapping protein_ID -> (List of) embedding(s), each shape (L, embed_dim) for each layer
    checkpoint (str): The pLM used for creating protein embeddings
    layers (List[int]): The hidden layers specified to be used for pooling. Rest is ignored. If no layers are given, uses last hidden layer.
    strategy (str): Pooling strategy 

    Returns
    -----------
    Dict[str, np.ndarray]: Dictionary mapping protein_ID -> pooled embedding vector (np.ndarray)
    """
    assert checkpoint in ("esm2_150m", "esm2_650m", "esm2_3b", "prot_t5", "esmc_300m", "esmc_600m", "esm3_open", "prost_t5", "glm2_650m"), f"Unsupported checkpoint: {checkpoint}"
    assert strategy in ("mean", "max", "FC"), f"Unsupported pooling strategy: {strategy}"
    
    if checkpoint in ("prot_t5", "prost_t5"):
        max_layer = 23
        embed_dim = 1024
    elif checkpoint == "esmc_300m":
        max_layer = 29
        embed_dim = 960
    elif checkpoint == "esmc_600m":
        max_layer = 35
        embed_dim = 1152
    elif checkpoint == "esm3_open":
        max_layer = 47
        embed_dim = 1536
    elif checkpoint == "glm2_650m":
        max_layer = 29
        embed_dim = 1280
    elif checkpoint == "esm2_150m":
        max_layer = 29
        embed_dim = 640
    elif checkpoint == "esm2_650m":
        max_layer = 32
        embed_dim = 1280
    elif checkpoint == "esm2_3b":
        max_layer = 35
        embed_dim = 2560

    pooled_embeddings = {}

    # set up attention pooling for fully connected layer strategy
    pooler = AttentionPool(emb_dim=embed_dim)
    pooler.eval()

    for pid, all_layers in tqdm(embeddings.items(), total=len(embeddings), desc="Pooling embeddings..."):
        # single-layer case
        if isinstance(all_layers, np.ndarray):
            max_layer = 0
            all_layers_list = [all_layers]
            logger.warning(f"Using only available layer for pooling, ignoring provided layers list.")
        else:
            # multiple layers
            all_layers_list = all_layers
            for l in layers:
                assert 0 <= l <= max_layer, f"Layer {l} out of range for {checkpoint} (max={max_layer})"

        # take last (or only) hidden layer if list is empty
        if len(layers) == 0:
            layers = [max_layer] 

        logger.info(f"{pid} Number of layers: {len(all_layers_list)}")
        for idx, layer in enumerate(all_layers_list):
            logger.info(f"layer {idx}: shape = {layer.shape}")
        
        extracted_layers = [all_layers_list[i] for i in layers] # each shape (L, embed_dim)
        logger.info(f"{pid} Number of extracted layers: {len(extracted_layers)}")

        if strategy == "mean":
            # 1) pool over sequence length 
            layer_vectors = [np.mean(layer, axis=0) for layer in extracted_layers]
            # 2) pool over layers
            final_vector = np.mean(np.stack(layer_vectors), axis=0)
        elif strategy == "max":
            # 1) pool over sequence length 
            layer_vectors = [np.max(layer, axis=0) for layer in extracted_layers]
            # 2) pool over layers
            final_vector = np.max(np.stack(layer_vectors), axis=0)
        else:
            with torch.no_grad():
                # 1) pool over sequence length
                layer_vectors = []
                for layer in extracted_layers:
                    layer = torch.from_numpy(layer).float()
                    layer_vectors.append(pooler(layer))

                # 2) pool over layers
                final_vector = torch.stack(layer_vectors).mean(dim=0).detach().cpu().numpy()

        pooled_embeddings[pid] = final_vector
        logger.info(f"Shape of final pooled vector for {pid}: {final_vector.shape}")

    return pooled_embeddings