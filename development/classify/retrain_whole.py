from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import set_determinism, read_hdf5, log, get_output_dir
from src.classify.data import encode_category, add_features, ProteinDataset, GenomicContextDataset, analyze_encoder, filter_unknowns, filter_cluster_representatives, filter_nan_embeddings, build_final_loader
from src.classify.model import MLP, CNN, ContextTransformer, CNN_MLP
from src.classify.train import fit_final_model

OmegaConf.register_new_resolver("get_output_dir", get_output_dir)

@hydra.main(config_path="./", config_name="config", version_base=None)

def main(cfg:DictConfig) -> None: #noqa: D401
    """
    Main function that performs classification of (un-/)pooled protein sequence embeddings.

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
    torch.set_float32_matmul_precision("high")
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if cfg.train.device == "cuda":
            logger.info("CUDA not available, using CPU.")

    # Read embeddings file, optionally specify metadata columns to extract
    embeddings_huge, metadata_huge = read_hdf5(cfg.data.embedding)

    # include features
    feats_list = None
    if cfg.features.bool:
        # prepare features 
        metadata_huge, feats_list = add_features(cfg.features.path, cfg.features.top_n, metadata_huge)

    # Filter out embeddings that contain only nans
    embeddings_huge, metadata_huge = filter_nan_embeddings(embeddings_huge, metadata_huge)

    # Filter cluster representatives
    colname = cfg.data.colname
    embeddings_filtered, metadata_filtered = filter_cluster_representatives(cfg.cluster.representatives, embeddings_huge, metadata_huge, colname)

    # Filter unknowns
    embeddings_unknowns, metadata_unknowns, embeddings_knowns, metadata_knowns = filter_unknowns(embeddings_filtered, metadata_filtered, colname)

    # Encode (sub)category
    map_colname = cfg.data.map_colname
    colname_top = cfg.data.colname_top
    map_colname_top = cfg.data.map_colname_top
    metadata_knowns_encoded, LE_SC = encode_category(metadata_knowns, colname, colname_top, map_colname, map_colname_top, cfg.cluster.cat)

    if cfg.context.bool:
        ds = GenomicContextDataset(
            embeddings=embeddings_filtered,
            metadata=metadata_filtered,
            embeddings_knowns=embeddings_knowns,
            metadata_knowns=metadata_knowns_encoded,
            context_size=cfg.context.size,
            embed_dim=cfg.data.embed_dim,
            colname=colname
        )

    else:
        ds = ProteinDataset(
            embeddings=embeddings_knowns,
            metadata=metadata_knowns_encoded,
            colname=colname,
            features=feats_list
        )

    # include bacterial and viral proteins
    if cfg.bac_vir.bool:
        embs_bac_vir, metadata_bac_vir = read_hdf5(cfg.bac_vir.embedded)
        embs_bac_vir, metadata_bac_vir = filter_nan_embeddings(embs_bac_vir, metadata_bac_vir)
        _, _, embs_bac_vir, metadata_bac_vir = filter_unknowns(embs_bac_vir, metadata_bac_vir, colname)

        metadata_bac_vir[f"{colname}_encoded"] = LE_SC.transform(metadata_bac_vir[colname])

        embeddings_combined = {**embeddings_knowns, **embs_bac_vir}
        metadata_combined = pd.concat([metadata_knowns_encoded, metadata_bac_vir], ignore_index=True)

        ds = ProteinDataset(
            embeddings=embeddings_combined,
            metadata=metadata_combined,
            colname=colname,
            features=None
        )

        logger.info("Included bac & vir:")
        logger.info(len(embs_bac_vir))
        logger.info(len(metadata_bac_vir))

        # get number of classes and calculate label weights
        num_categories, label_map, label_weights = analyze_encoder(LE_SC, metadata_combined, colname)
    
    else:
        num_categories, label_map, label_weights = analyze_encoder(LE_SC, metadata_knowns_encoded, colname)

    loader = build_final_loader(ds, batch_size=cfg.optim.batch_size, context=cfg.context.bool, num_workers=cfg.data.num_workers)
    
    # Loss function with class weights to handle imbalance
    weights = torch.tensor(label_weights.values, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    # Persist artifacts
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    if cfg.data.pooled:
        if cfg.context.bool:
            models = {}

            for mod, model_cfg in cfg.models.items():
                
                model_context = ContextTransformer(
                    num_heads=model_cfg.context.num_heads,
                    dim_feedforward=model_cfg.context.dim_feedforward,
                    num_layers=model_cfg.context.num_layers,
                    dropout=model_cfg.context.dropout,
                    num_classes=num_categories,
                    embed_dim=cfg.data.embed_dim,
                    context_size=cfg.context.size
                ).to(device)
                models[f"context_{mod}"] = model_context
                break

        else:
            models = {}

            for mod, model_cfg in cfg.models.items():
                model = MLP(
                    trial=None,
                    in_dim=cfg.data.embed_dim,
                    num_dimensions=model_cfg.mlp.num_dimensions,
                    num_neurons=list(model_cfg.mlp.num_neurons),
                    dropout=model_cfg.mlp.dropout,
                    num_classes=num_categories,
                ).to(device)
                models[f"mlp_{mod}"] = model
                break
    else:
        models = {}

        for mod, model_cfg in cfg.models.items():
            if model_cfg.cnn.bool:
                model_cnn = CNN(
                    trial=None,
                    in_channels=cfg.data.embed_dim,
                    num_conv_layers=model_cfg.cnn.num_conv_layers,
                    num_filters=list(model_cfg.cnn.num_filters),
                    kernel_sizes=list(model_cfg.cnn.kernel_sizes),
                    dropout=model_cfg.cnn.dropout,
                    num_classes=num_categories,
                    dilations=list(model_cfg.cnn.dilations),
                    use_dilation=model_cfg.cnn.use_dilation,
                    mean_max=model_cfg.cnn.mean_max,
                    n_feats=0 if cfg.features.bool != True else cfg.features.top_n,
                    use_linear_attention=model_cfg.cnn.use_linear_attention,
                    use_nonlinear_attention=model_cfg.cnn.use_nonlinear_attention,
                ).to(device)
                models[f"cnn_{mod}"] = model_cnn
            
            if model_cfg.cnn_mlp.bool:
                model_cnn_mlp = CNN_MLP(
                    trial=None,
                    in_channels=cfg.data.embed_dim,
                    num_conv_layers=model_cfg.cnn_mlp.num_conv_layers,
                    num_filters=list(model_cfg.cnn_mlp.num_filters),
                    kernel_sizes=list(model_cfg.cnn_mlp.kernel_sizes),
                    dropout_conv=model_cfg.cnn_mlp.dropout_conv,
                    num_classes=num_categories,
                    dilations=list(model_cfg.cnn_mlp.dilations),
                    use_dilation=model_cfg.cnn_mlp.use_dilation,
                    use_linear_attention=model_cfg.cnn_mlp.use_linear_attention,
                    use_nonlinear_attention=model_cfg.cnn_mlp.use_nonlinear_attention,
                    num_dimensions=model_cfg.cnn_mlp.num_dimensions,
                    num_neurons=model_cfg.cnn_mlp.num_neurons,
                    dropout_mlp=model_cfg.cnn_mlp.dropout_mlp,
                ).to(device)
                models[f"cnn_mlp_{mod}"] = model_cnn_mlp
       

    
    for model_key, model in models.items():
        logger.info(f"Model configuration key: {model_key}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        logger.info("Model architecture:\n%s", repr(model))


        assert cfg.optim.optimizer in ("Adam", "AdamW", "SGD"), "optimizer not supported"
        if cfg.optim.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
        elif cfg.optim.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.optim.lr,
                weight_decay=cfg.optim.weight_decay,
            )

        logger.info(
            "Config (core): batch_size=%d lr=%.3g weight_decay=%.3g epochs=%d device=%s",
            cfg.optim.batch_size,
            cfg.optim.lr,
            cfg.optim.weight_decay,
            cfg.optim.epochs,
            device,
        )

        logger.info(
            "Estimated steps/epoch: %d (len(loader)) | total updates ~ %d",
            len(loader),
            len(loader) * cfg.optim.epochs,
        )

        logger.info(f"Optimizer: {cfg.optim.optimizer}(lr=%.3g, weight_decay=%.3g)", cfg.optim.lr, cfg.optim.weight_decay)
        logger.info("Loss: CrossEntropyLoss | Starting training loop ...")
        
        # Set up logging
        log_fn = log(logger)

        fit_final_model(
            model=model,
            loader=loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg.optim.epochs,
            device=device,
            log_fn=log_fn,
            log_interval=cfg.train.log_interval
        )

        model_path = os.path.join(output_dir, f"{model_key}_final.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
                "label_map": label_map,
            },
            model_path,
        )
        logger.info(f"Saved final model checkpoint to {model_path}")



if __name__ == "__main__":
    main()

