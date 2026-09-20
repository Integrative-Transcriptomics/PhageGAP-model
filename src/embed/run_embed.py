from __future__ import annotations
import torch
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os

from src.utils import set_determinism, create_embeddings_file
from src.embed.embed import instantiate_model, load_df, preprocess_df, compute_embeddings

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
                    
@hydra.main(config_path="./", config_name="config", version_base=None)

def main(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that creates embedings for protein sequences.

    Parameters
    -----------
    cfg (DictConfig): The Hydra configuration object
    """
    logger = logging.getLogger(__name__)
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # 1. Set determinism and environment
    set_determinism(cfg.seed)
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    else:
        device = torch.device("cpu")
        if cfg.device == "cuda":
            logger.info("CUDA not available, using CPU.")

    # 2. Instantiate model
    checkpoint = cfg.embeddings.model_type
    logger.info(f"Instantiating model: {checkpoint} on {device}")
    model, tokenizer = instantiate_model(checkpoint, device)

    # 3. Preprocess amino acid sequences
    df = load_df(cfg.data.metadata_tsv)
    preprocessed_df = preprocess_df(df, checkpoint)

    # 4. Compute embeddings
    embed_dict = compute_embeddings(checkpoint, preprocessed_df, model, tokenizer, cfg.embeddings.only_last)
   
    # 5. Create and store embeddings 
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]
    embeddings_path = os.path.join(output_dir, f"protein_embeddings_last_{str(cfg.embeddings.only_last)}_{cfg.embeddings.model_type}.h5")
    create_embeddings_file(embed_dict, preprocessed_df, embeddings_path)
    logger.info(f"Saved embeddings to {embeddings_path}")


if __name__ == "__main__":
    main()
