from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import set_determinism, read_hdf5, get_checkpoint_from_embedding_filename, create_embeddings_file
from src.pool.pool import pool_embeddings, filter_nan_embeddings

@hydra.main(config_path="./", config_name="config", version_base=None)

def main(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that performs pooling of protein sequence embeddings.

    Parameters
    -----------
    cfg: DictConfig
        The Hydra configuration object, composed from YAML files and command-line
        overrides. It contains all settings for the run.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Set determinism and environment
    set_determinism(cfg.seed)

    # 2. Read embeddings file, optionally specify metadata columns to extract
    embeddings, metadata = read_hdf5(cfg.data.embedding)
    embeddings, metadata = filter_nan_embeddings(embeddings, metadata)

    # 3. Pool embeddings
    checkpoint = get_checkpoint_from_embedding_filename(cfg.data.embedding)
    pooled_embeddings = pool_embeddings(embeddings, checkpoint, layers=cfg.pool.layers, strategy=cfg.pool.strategy)

    # 4. Store pooled embeddings 
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]
    embeddings_path = os.path.join(output_dir, f"protein_embeddings_{checkpoint}_{"-".join(str(l) for l in cfg.pool.layers)}_{cfg.pool.strategy}.h5")
    create_embeddings_file(pooled_embeddings, metadata, str(embeddings_path))
    logger.info(f"Saved pooled embeddings to {embeddings_path}")


if __name__ == "__main__":
    main()

