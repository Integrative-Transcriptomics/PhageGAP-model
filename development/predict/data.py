"""Data utilities"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
import torch
import sys
import pandas as pd
import re
from src.classify.model import MLP, CNN
from Bio import SeqIO
from torch.nn.utils.rnn import pad_sequence


def load_fasta(path_to_fasta):
    """Load fasta to make predictions for"""
    proteins = []
    with open(path_to_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            protein_data = {
                "protein_ID": record.id,
                "description": record.description,
                "protein_seq": str(record.seq)
            }
            proteins.append(protein_data)
    df = pd.DataFrame.from_dict(proteins)
    return df

def filter_unknowns(in_path, embeddings: Dict[str, np.ndarray], metadata_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Filters unknowns based on .csv in in_path"""
    df_unknowns = pd.read_csv(in_path)

    filtered_df = metadata_df[metadata_df.protein_ID.isin(df_unknowns["protein_ID"])]

    keep_ids = set(filtered_df["protein_ID"])
    embeddings_filtered = {k: v for k, v in embeddings.items() if k in keep_ids}
    return embeddings_filtered, filtered_df

def extract_model_number(path: str):
    """Extracts number of model used to make predictions"""
    match = re.search(r"model(\d+)\_final.pt$", path)
    if match:
        num = int(match.group(1))

    return num