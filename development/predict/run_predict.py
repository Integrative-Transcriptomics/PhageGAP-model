from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import read_hdf5, set_determinism, create_embeddings_file
from src.predict.data import filter_unknowns, extract_model_number, load_fasta
from src.predict.predict import predict
from src.embed.embed import instantiate_model, preprocess_df, compute_embeddings, monitor_model_loading
from src.pool.pool import pool_embeddings

@hydra.main(config_path="./", config_name="config", version_base=None)

def main(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that performs prediction of (un-/)pooled protein sequence embeddings.

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
    else:
        device = torch.device("cpu")
        if cfg.device == "cuda":
            logger.info("CUDA not available, using CPU.")

    # 2. Get model number
    model_no = extract_model_number(cfg.model.path)
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    if cfg.all.bool:
        embeddings_huge, metadata_huge = read_hdf5(cfg.unknowns.embedding)

        # Predict
        predictions_df, filter_dict = predict(embeddings_huge, metadata_huge, cfg.model.path, model_no, cfg.model.type)
        df = metadata_huge.copy()


    elif cfg.unknowns.bool:
        # Read embeddings file, optionally specify metadata columns to extract
        embeddings_huge, metadata_huge = read_hdf5(cfg.unknowns.embedding)

        # Filter unknowns
        embeddings_unknown, metadata_unknown = filter_unknowns(cfg.unknowns.ids, embeddings_huge, metadata_huge)

        # Predict
        predictions_df, filter_dict = predict(embeddings_unknown, metadata_unknown, cfg.model.path, model_no, cfg.model.type)
        df = metadata_unknown.copy()

    else:
        # Instantiate model
        checkpoint = cfg.embeddings.model_type
        logger.info(f"Instantiating model: {checkpoint} on {device}")
        model, tokenizer = monitor_model_loading(checkpoint, device)

        # Preprocess amino acid sequences
        df = load_fasta(cfg.new_fasta.path)
        preprocessed_df = preprocess_df(df, checkpoint)

        # Compute embeddings
        embed_dict = compute_embeddings(checkpoint, preprocessed_df, model, tokenizer, cfg.embeddings.only_last)

        embed_dict_pooled = pool_embeddings(embed_dict, checkpoint, layers=cfg.pool.layers, strategy=cfg.pool.strategy)

        if cfg.model.type == "mlp":
        # Pool embeddings
            embed_dict = embed_dict_pooled
    
        # Predict
        predictions_df, filter_dict = predict(embed_dict, preprocessed_df, cfg.model.path, model_no, cfg.model.type)

        embeddings_path = os.path.join(output_dir, f"protein_embeddings_{checkpoint}_{"-".join(str(l) for l in cfg.pool.layers)}_{cfg.pool.strategy}.h5")
        create_embeddings_file(embed_dict_pooled, preprocessed_df, str(embeddings_path))
        logger.info(f"Saved pooled embeddings to {embeddings_path}")

    # Store
    predictions_df.to_csv(os.path.join(output_dir, "predictions.tsv"), sep="\t", index=False)

    cnn_embeddings_path = os.path.join(output_dir, "cnn_embeddings.h5")
    create_embeddings_file(filter_dict, df, str(cnn_embeddings_path))
    logger.info(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    main()

