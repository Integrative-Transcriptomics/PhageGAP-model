from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import set_determinism, read_hdf5, get_checkpoint_from_embedding_filename
from src.prepare_for_webapp.prepare import perform_pca, perform_tsne
from src.classify.data import filter_cluster_representatives, filter_unknowns, filter_nan_embeddings
from src.pool.pool import pool_embeddings

@hydra.main(config_path="./", config_name="config", version_base=None)

def main(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that performs preparation of pooled protein sequence embeddings.

    Parameters
    -----------
    cfg: DictConfig
        The Hydra configuration object, composed from YAML files and command-line
        overrides. It contains all settings for the run.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # Set determinism and environment
    set_determinism(cfg.seed)

    # Read embeddings file, optionally specify metadata columns to extract
    embeddings_huge, metadata_huge = read_hdf5(cfg.data.embedding)
    embeddings_huge, metadata_huge = filter_nan_embeddings(embeddings_huge, metadata_huge)

    # filter cluster representatives
    embeddings_filtered, metadata_filtered = filter_cluster_representatives(cfg.cluster.representatives, embeddings_huge, metadata_huge, colname=cfg.mapping.colname)

    # include bacterial and viral proteins
    if cfg.bac_vir.bool:
        embs_bac_vir, metadata_bac_vir = read_hdf5(cfg.bac_vir.embedded)
        embs_bac_vir, metadata_bac_vir = filter_nan_embeddings(embs_bac_vir, metadata_bac_vir)
        embeddings_filtered = embeddings_filtered | embs_bac_vir
        metadata_filtered["dataset"] = "PhageOnly"
        metadata_bac_vir["dataset"] = "BacVir"
        metadata_filtered = pd.concat([metadata_filtered, metadata_bac_vir], axis=0)

    # filter for only labeled proteins
    embeddings_unknowns, metadata_unknowns, embeddings_knowns, metadata_knowns = filter_unknowns(embeddings_filtered, metadata_filtered, cfg.mapping.colname)

    if cfg.pool.bool:
        # Pool embeddings
        checkpoint = get_checkpoint_from_embedding_filename(cfg.data.embedding)
        pooled_embeddings = pool_embeddings(embeddings_knowns, checkpoint, layers=cfg.pool.layers, strategy=cfg.pool.strategy)
        pca, coords, ids = perform_pca(pooled_embeddings, cfg.pca.type, cfg.pca.n_components, cfg.seed)
    else:
        # Perform PCA
        pca, coords, ids = perform_pca(embeddings_knowns, cfg.pca.type, cfg.pca.n_components, cfg.seed)

    # Perform T-SNE
    X_tsne = perform_tsne(coords, cfg.tsne.perplexity, cfg.tsne.metric, cfg.seed)

    # Save everything in output_dir
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    # PCA bundle
    pca_bundle = {"model": pca, "coords": coords, "ids": ids}
    pca_path = os.path.join(output_dir, "pca.joblib")
    joblib.dump(pca_bundle, pca_path)
    logger.info(f"Saved PCA bundle to {pca_path}")

    # t-SNE bundle
    tsne_bundle = {"coords": X_tsne, "ids": ids}
    tsne_path = os.path.join(output_dir, "tsne.joblib")
    joblib.dump(tsne_bundle, tsne_path)
    logger.info(f"Saved tSNE bundle to {tsne_path}")

    metadata_path = os.path.join(output_dir, "metadata.tsv")
    metadata_knowns.to_csv(metadata_path, sep="\t", index=False)
    logger.info(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    main()

