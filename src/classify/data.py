"""Classification utilities"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

logger = logging.getLogger(__name__)

def add_features(in_path: str, num_feats: int, metadata_df: pd.DataFrame):
    """Adds features to the metadata DataFrame. Assumes 'protein_ID' is in the first column and features are sorted."""
    assert 0 <= num_feats <= 30, "Number of features must lie in [0 - 30]"
    feats_df = pd.read_csv(in_path, sep="\t")

    feature_cols = feats_df.columns[1 : num_feats + 1]
    first_n_feats = feats_df[["protein_ID", *feature_cols]]

    added_df = pd.merge(metadata_df, first_n_feats, how="inner", on="protein_ID")

    return added_df, feature_cols.tolist()


def filter_cluster_representatives(in_path: str, embeddings: Dict[str, np.ndarray], metadata_df: pd.DataFrame, colname: str) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Filters embeddings and metadata_df, keeps only those that are present in .tsv from in_path"""
    repr_df = pd.read_csv(in_path, sep="\t")
    cluster_repr = repr_df["protein_ID"].unique()

    metadata_filtered = metadata_df[metadata_df["protein_ID"].isin(cluster_repr)]

    if "assigned_cat" not in metadata_filtered.columns:
        # get category columns from repr_df
        cat_cols = ["protein_ID", "assigned_subcat", "assigned_cat", "assigned_top"]
        repr_cat = repr_df[cat_cols]

        # add to metadata_filtered
        metadata_filtered = metadata_filtered.merge(repr_cat, on="protein_ID", how="left")

    embeddings_filtered = {k: v for k, v in embeddings.items() if k in cluster_repr}
    metadata_filtered = clean_df(metadata_filtered, colname)
    
    return embeddings_filtered, metadata_filtered


def clean_df(df, colname):
    """Cleanes column 'colname' from df by setting to 'Unknown' if entry contains ';'"""
    mask = df[colname].str.contains(";")
    df.loc[mask, colname] = "Unknown"
    return df


def filter_nan_embeddings(
        embeddings: Dict[str, np.ndarray | List[np.ndarray]],
        metadata_df: pd.DataFrame
        ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
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


def filter_unknowns(
        embeddings: Dict[str, np.ndarray | List[np.ndarray]],
        metadata_df: pd.DataFrame,
        colname: str
        ) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Filters embeddings and metadata_df, keeps only those where column 'colname' is not 'Unknown'"""
    assert colname in ("assigned_subcat", "assigned_cat", "assigned_top"), "colname must be one of ('assigned_subcat', 'assigned_cat', 'assigned_top')"

    metadata_df = clean_df(metadata_df, colname)
    unknowns_df = metadata_df[metadata_df[colname] == "Unknown"]
    knowns_df = metadata_df[metadata_df[colname] != "Unknown"]

    unknowns_ids = set(unknowns_df["protein_ID"])
    embeddings_unknowns = {k: v for k, v in embeddings.items() if k in unknowns_ids}

    knowns_ids = set(knowns_df["protein_ID"])
    embeddings_knowns = {k: v for k, v in embeddings.items() if k in knowns_ids}

    return embeddings_unknowns, unknowns_df, embeddings_knowns, knowns_df
    

def encode_category(metadata: pd.DataFrame, colname: str, colname_top: str, map_colname: str, 
                    map_colname_top: str, map_path: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encodes column 'colname' of metadata DataFrame. Returns metadata and LabelEncoder"""
    assert colname_top in ("assigned_cat", "assigned_top"), "colname_top must be one of ('assigned_cat', 'assigned_top')"
    assert map_colname in ("Subcategory", "Category", "Top_Category"), "map_colname must be one of ('Subcategory', 'Category', 'Top_Category')"
    assert map_colname_top in ("Category", "Top_Category"), "map_colname_top must be one of ('Category', 'Top_Category')"

    metadata = metadata.copy()

    keywords_map = pd.read_csv(map_path)

    if colname != "assigned_top":
        CATEGORY_TO_TOP = keywords_map[[map_colname, map_colname_top]].drop_duplicates().set_index(map_colname)[map_colname_top].to_dict()
        metadata[colname_top] = metadata[colname].map(CATEGORY_TO_TOP)

    elif colname == "assigned_top":
        colname_top = "assigned_top2"
        metadata[colname_top] = metadata[colname]

    # order such that lower-level categories belonging to the same higher-level category get consecutive numbers
    ordered_subcategories = metadata[[colname, colname_top]].drop_duplicates().sort_values([colname_top, colname])[colname].to_list()
    le = LabelEncoder()
    le.classes_ = np.array(ordered_subcategories)

    # Transform
    metadata[f"{colname}_encoded"] = le.transform(metadata[colname])
    return metadata, le


def analyze_encoder(Encoder: LabelEncoder, metadata_encoded: pd.DataFrame, colname: str) -> Tuple[int, Dict[str, int], pd.Series]:
    """Analyzes Encoder.

    Returns:
        num_categories (int): Number of Encoder classes
        label_map (Dict[str, int]): Dictionary mapping class (str) to encoding (int)
        label_weights (pd.Series): Weights for each class computed as the inverse of class frequency, normalized to have mean 1.
                        Used in loss function to give more importance to rare classes and less to frequent ones.
        """
    num_categories = len(Encoder.classes_)
    logger.info(f"Number of categories: {num_categories}")
    label_map = dict(zip(Encoder.classes_, Encoder.transform(Encoder.classes_)))
    label_counts = metadata_encoded[f"{colname}_encoded"].value_counts()

    logger.info(f"Labels: {Encoder.classes_}")
    logger.info(f"Label counts: {label_counts}")

    label_weights = 1.0 / label_counts # inverse of class frequency
    label_weights = label_weights / label_weights.mean() # normalized
    logger.info(f"Label weights: {label_weights}")

    return num_categories, label_map, label_weights

class ProteinDataset(Dataset):
    """Dataset of protein embeddings"""
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray], 
        metadata: pd.DataFrame,
        colname: str,
        features: List[str] | None
    ) -> None:
        self.embeddings = embeddings
        self.metadata = metadata
        self.proteins = sorted(embeddings.keys())
        self.samples: List[Tuple[str, torch.Tensor, int, int, str]] = []  # (protein_ID, embedding, cat_encoded, length, domain)
        self.id_to_label = dict(zip(metadata["protein_ID"], metadata[f"{colname}_encoded"]))
        if features is not None:
            self.id_to_features = {pid: row[features].values.astype(np.float32) for pid, row in metadata.set_index("protein_ID").iterrows()}
        else:
            self.id_to_features = {}
        if "table_origin" in metadata.columns:
            self.id_to_domain = {pid: (origin if origin in ["bac", "vir"] else "phg")
                                 for pid, origin in zip(metadata["protein_ID"], metadata["table_origin"])}
        else: 
            self.id_to_domain = {pid: "phg" for pid in metadata["protein_ID"]}

        for protein_ID, embedding in self.embeddings.items():
            label = self.id_to_label[protein_ID]
            embedding = torch.from_numpy(embedding).float()
            length = embedding.shape[0]
            domain = self.id_to_domain[protein_ID]
            if features is not None:
                feats = self.id_to_features[protein_ID]  # gives back list
                feats = torch.tensor(feats, dtype=torch.float32)
            else:
                feats = None

            self.samples.append((protein_ID, embedding, label, length, domain, feats))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        protein_ID, embedding, assigned_cat_encoded, length, domain, feats = self.samples[idx]
        return embedding, torch.tensor(assigned_cat_encoded, dtype=torch.long), length, protein_ID, domain, feats
    

class GenomicContextDataset(Dataset):
    """Dataset of protein embeddings"""

    def __init__(
        self,
        embeddings: Dict[str, np.ndarray | List[np.ndarray]], 
        metadata: pd.DataFrame,
        embeddings_knowns: Dict[str, np.ndarray | List[np.ndarray]], 
        metadata_knowns: pd.DataFrame,
        context_size: int,
        embed_dim: int,
        colname: str
    ) -> None:
        self.embeddings = embeddings
        self.context_size = context_size
        self.embed_dim = embed_dim

        self.protein_ids = list(embeddings_knowns.keys())
        self.id_to_label = dict(zip(metadata_knowns["protein_ID"], metadata_knowns[f"{colname}_encoded"]))

        self.zero_embedding = torch.zeros(embed_dim)

        # protein order
        self.phage_to_proteins = metadata.sort_values("protein_nr").groupby("phage_ID")["protein_ID"].apply(list).to_dict()

        # protein index lookup
        self.protein_to_index = {}
        for phage, proteins in self.phage_to_proteins.items():
            for i, protein in enumerate(proteins):
                self.protein_to_index[protein] = (phage, i)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, idx: int):
        protein_ID = self.protein_ids[idx]
        label = self.id_to_label[protein_ID]

        phage, center_idx = self.protein_to_index[protein_ID]
        proteins = self.phage_to_proteins[phage]

        half = self.context_size // 2

        embeddings_list = []
        mask = []

        for i in range(center_idx-half, center_idx+half+1):
            if i < 0 or i >= len(proteins):
                emb = self.zero_embedding.clone()
                mask.append(False)
            else:
                pid = proteins[i]
                arr = self.embeddings.get(pid)

                if arr is None:
                    emb = self.zero_embedding.clone()
                    mask.append(False)
                else:
                    emb = torch.from_numpy(arr).float()
                    mask.append(True)

            embeddings_list.append(emb)

        context_embedding = torch.stack(embeddings_list)
        mask = torch.tensor(mask)

        return context_embedding, torch.tensor(label, dtype=torch.long), mask, protein_ID

def build_cv_folds(trainval_proteins, trainval_labels, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []

    for train_idx, val_idx in sgkf.split(X=trainval_proteins, y=trainval_labels, groups=trainval_proteins):
        train_pids = trainval_proteins[train_idx]
        val_pids = trainval_proteins[val_idx]
        folds.append((train_pids, val_pids))

    return folds


def build_fold_assignments(trainval_proteins, trainval_labels, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_assignments = {}

    for fold_id, (_, val_idx) in enumerate(sgkf.split(X=trainval_proteins, y=trainval_labels,  groups=trainval_proteins)):
        for p in trainval_proteins[val_idx]:
            fold_assignments[p] = fold_id

    return fold_assignments


def build_dataloaders_cv(
    dataset: Dataset,
    batch_size: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    num_workers: int,
    n_folds: int = 5
) -> Tuple[DataLoader, List[Tuple[DataLoader, DataLoader]]]:
    """Create train/val/test loaders derived deterministically from shuffled protein order using
    val_fraction / test_fraction proportions stratified for label (rounded, ensuring >=1 each).
    """
    protein_for_index = [dataset.samples[i][0] for i in range(len(dataset.samples))]

    id_to_label = dataset.id_to_label
    all_proteins = np.array(sorted(set(protein_for_index)))
    all_labels = np.array([id_to_label[p] for p in all_proteins])

    def get_indices(protein_set):
        protein_set = set(protein_set)
        return [i for i, p in enumerate(protein_for_index) if p in protein_set]

    # Stratified Sampling for train and val
    train_pids, val_test_pids, train_lab, val_test_lab = train_test_split(all_proteins, all_labels,
                                                test_size=val_fraction+test_fraction,
                                                random_state=seed,
                                                stratify=all_labels)


    test_ratio = test_fraction / (val_fraction + test_fraction)

    val_pids, test_pids, val_lab, test_lab = train_test_split(val_test_pids, val_test_lab,
                                            test_size=test_ratio,
                                            random_state=seed,
                                            stratify=val_test_lab)

    
    train_val_pids = np.array(sorted(set(train_pids).union(set(val_pids))))
    train_val_labels = np.array([id_to_label[p] for p in train_val_pids])
    fold_assignments = build_fold_assignments(train_val_pids, train_val_labels, n_splits=n_folds, seed=seed)

    test_idx = get_indices(test_pids)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    cv_folds = []

    for fold in range(n_folds):

        cv_train_pids = [p for p in train_val_pids if fold_assignments[p] != fold]
        cv_val_pids = [p for p in train_val_pids if fold_assignments[p] == fold]

        train_idx = get_indices(cv_train_pids)
        val_idx = get_indices(cv_val_pids)

        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

        cv_folds.append((train_loader, val_loader))

    return test_loader, cv_folds



def build_dataloaders(
    dataset: Dataset,
    batch_size: int,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    context: bool,
    num_workers: int,
    optimize: bool
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create train/val/test loaders derived deterministically from shuffled protein order using
    val_fraction / test_fraction proportions stratified for label (rounded, ensuring >=1 each).
    """
    if context is True:
        protein_for_index = [dataset.protein_ids[i][0] for i in range(len(dataset.protein_ids))]
    else:
        protein_for_index = [dataset.samples[i][0] for i in range(len(dataset.samples))]
        id_to_domain = dataset.id_to_domain

    id_to_label = dataset.id_to_label
    all_proteins = np.array(sorted(set(protein_for_index)))
    all_labels = np.array([id_to_label[p] for p in all_proteins])

    def get_indices(protein_set):
        protein_set = set(protein_set)
        return [i for i, p in enumerate(protein_for_index) if p in protein_set]

    # Stratified Sampling for train and val
    train_pids, val_test_pids, train_lab, val_test_lab = train_test_split(all_proteins, all_labels,
                                                test_size=val_fraction+test_fraction,
                                                random_state=seed,
                                                stratify=all_labels)

    if not optimize:
        test_ratio = test_fraction / (val_fraction + test_fraction)

        val_pids, test_pids, val_lab, test_lab = train_test_split(val_test_pids, val_test_lab,
                                                test_size=test_ratio,
                                                random_state=seed,
                                                stratify=val_test_lab)

        train_idx = get_indices(train_pids)
        val_idx = get_indices(val_pids)
        test_idx = get_indices(test_pids)

        assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0, "Empty split subset encountered"

        val_ds = torch.utils.data.Subset(dataset, val_idx)

    else:
        train_idx = get_indices(train_pids)
        test_idx = get_indices(val_test_pids) # merge val and test together for optuna
        val_ds = None


    train_ds = torch.utils.data.Subset(dataset, train_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    if context is True:
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_context),
            DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_context) if val_ds else None,
            DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_context),
        )

    else:
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn),
            DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn) if val_ds else None,
            DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn),
        )


def build_final_loader(
    dataset: Dataset,
    batch_size: int,
    context: bool,
    num_workers: int
) -> DataLoader:
    """Create final loader containing all known labeled data."""
    if context:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_context)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)


def collate_fn(batch):
    """Collates a batch by padding to max length"""
    embeddings, labels, lengths, ids, domains, feats_tensor = zip(*batch)

    if feats_tensor[0] is None:
        feats = None
    else:
        feats = torch.stack(feats_tensor)

    # pooled embeddings
    if embeddings[0].ndim == 1:
        x = torch.stack(embeddings)
        y = torch.tensor(labels, dtype=torch.long)
        mask = None
        return x, y, list(ids), mask, feats

    # per-residue embeddings
    max_len = max(lengths)
    D = embeddings[0].shape[1]

    padded = torch.zeros(len(batch), max_len, D)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (emb, L) in enumerate(zip(embeddings, lengths)):
        padded[i, :L] = emb
        mask[i, :L] = True

    return padded, torch.tensor(labels, dtype=torch.long), list(ids), mask, feats


def collate_context(batch):
    embeddings, labels, masks, ids = zip(*batch)
    x = torch.stack(embeddings) # (batch, context_size, embed_dim)
    y = torch.tensor(labels, dtype=torch.long)
    mask = torch.stack(masks)

    return x, y, list(ids), mask