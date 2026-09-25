from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
import torch
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd

# Optuna hyperparameter optimization framework adapted from 
# https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py
# Thank you for publishing!

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import get_output_dir, set_determinism, read_hdf5, log, save_json
from src.classify.data import encode_category, add_features, ProteinDataset, analyze_encoder, build_dataloaders_cv, collate_fn, filter_unknowns, filter_cluster_representatives, filter_nan_embeddings
from src.classify.model import MLP, CNN_MLP, CNN, CNN_Flattened
from src.classify.train import fit_cv

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
    # Persist artifacts
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Set up logging
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
    logger.info("All:")
    logger.info(len(embeddings_huge))
    logger.info(len(metadata_huge))

    # include features
    feats_list = None
    if cfg.features.bool:
        # prepare features 
        metadata_huge, feats_list = add_features(cfg.features.path, cfg.features.top_n, metadata_huge)

    # Filter out embeddings that contain only np.nan
    embeddings_huge, metadata_huge = filter_nan_embeddings(embeddings_huge, metadata_huge)
    logger.info("Filter out nan:")
    logger.info(len(embeddings_huge))
    logger.info(len(metadata_huge))

    # Filter cluster representatives
    colname = cfg.data.colname
    embeddings_filtered, metadata_filtered = filter_cluster_representatives(cfg.cluster.representatives, embeddings_huge, metadata_huge, colname)
    logger.info("Filter repr.:")
    logger.info(len(embeddings_filtered))
    logger.info(len(metadata_filtered))

    # Filter unknowns
    logger.info(metadata_filtered[colname].unique())
    embeddings_unknowns, metadata_unknowns, embeddings_knowns, metadata_knowns = filter_unknowns(embeddings_filtered, metadata_filtered, colname)

    logger.info("Filter unknowns:")
    logger.info(len(embeddings_knowns))
    logger.info(len(metadata_knowns))
    logger.info(metadata_knowns[colname].unique())

    # Encode (sub)category
    map_colname = cfg.data.map_colname
    colname_top = cfg.data.colname_top
    map_colname_top = cfg.data.map_colname_top
    metadata_knowns_encoded, LE_SC = encode_category(metadata_knowns, colname, colname_top, map_colname, map_colname_top, cfg.cluster.cat)
    print("Encoded:")
    logger.info(metadata_knowns_encoded[f"{colname}_encoded"].unique())

    ds = ProteinDataset(
        embeddings=embeddings_knowns,
        metadata=metadata_knowns_encoded,
        colname=colname,
        features=feats_list
    )

    # load bacterial and viral proteins
    embs_bac_vir, metadata_bac_vir = read_hdf5(cfg.bac_vir.embedded)
    embs_bac_vir, metadata_bac_vir = filter_nan_embeddings(embs_bac_vir, metadata_bac_vir)
    _, _, embs_bac_vir, metadata_bac_vir = filter_unknowns(embs_bac_vir, metadata_bac_vir, colname)

    metadata_bac_vir[f"{colname}_encoded"] = LE_SC.transform(metadata_bac_vir[colname])

    ds_bac_vir = ProteinDataset(
        embeddings=embs_bac_vir,
        metadata=metadata_bac_vir,
        colname=colname,
        features=None
    ) 

    metadata_combined = pd.concat([metadata_knowns_encoded, metadata_bac_vir], ignore_index=True)
    # get number of classes and calculate label weights
    num_categories, label_map, label_weights = analyze_encoder(LE_SC, metadata_combined, colname)

    # Loss function with class weights to handle imbalance
    logger.info(f"label_weights.values: {label_weights.values}")
    weights = torch.tensor(label_weights.values, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    test_loader, cv_folds = build_dataloaders_cv(dataset=ds, batch_size=cfg.optim.batch_size, val_fraction=cfg.data.val_size,
                                                 test_fraction=cfg.data.test_size, seed=cfg.seed, num_workers=cfg.data.num_workers, n_folds=5)
    
    best_epochs = []
    cv_results = []
    for fold_id, (train_loader, val_loader) in enumerate(cv_folds):
        logger.info(f"Fold {fold_id}")
        fold_dir = os.path.join(output_dir, f"fold_{fold_id+1}")
        os.makedirs(fold_dir, exist_ok=True)
       
        # include bac + vir
        train_dataset = ConcatDataset([train_loader.dataset, ds_bac_vir])

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.optim.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_fn
        )

        logger.info("Included bac & vir:")
        logger.info(len(embs_bac_vir))
        logger.info(len(metadata_bac_vir))

        # Pre-training diagnostics
        logger.info(
            "Dataset: total=%d | train=%d | val=%d | test=%d",
            len(ds),
            len(train_loader.dataset),  # type: ignore[arg-type]
            len(val_loader.dataset),  # type: ignore[arg-type]
            len(test_loader.dataset),  # type: ignore[arg-type]
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
            "Estimated steps/epoch: %d (len(train_loader)) | total updates ~ %d",
            len(train_loader),
            len(train_loader) * cfg.optim.epochs,
        )

        if cfg.data.pooled:
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

                if model_cfg.cnn_flattened.bool:
                    model_cnn = CNN_Flattened(
                        trial=None,
                        in_channels=cfg.data.embed_dim,
                        num_conv_layers=model_cfg.cnn_flattened.num_conv_layers,
                        num_filters=list(model_cfg.cnn_flattened.num_filters),
                        kernel_sizes=list(model_cfg.cnn_flattened.kernel_sizes),
                        dropout=model_cfg.cnn_flattened.dropout,
                        num_classes=num_categories,
                        dilations=list(model_cfg.cnn_flattened.dilations),
                        use_dilation=model_cfg.cnn_flattened.use_dilation,
                        mean_max=model_cfg.cnn_flattened.mean_max,
                        n_feats=0 if cfg.features.bool != True else cfg.features.top_n,
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

            logger.info(f"Optimizer: {cfg.optim.optimizer}(lr=%.3g, weight_decay=%.3g)", cfg.optim.lr, cfg.optim.weight_decay)
            logger.info("Loss: CrossEntropyLoss | Starting training loop ...")

            # Set up logging
            log_fn = log(logger)
            os.makedirs(os.path.join(fold_dir, "predictions", model_key), exist_ok=True)

            log_objects, best_epoch, best_val_f1 = fit_cv(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=cfg.optim.epochs,
                device=device,
                log_fn=log_fn,
                log_interval=cfg.train.log_interval,
                encoder=LE_SC,
                path=os.path.join(fold_dir, "predictions", model_key)
            )

            normal_dict_log_objects = {k: dict(v) for k, v in log_objects.items()}
            save_json(normal_dict_log_objects, os.path.join(fold_dir, f"log_objects_{model_key}.json"))
            cv_results = {"best_epoch": best_epoch, "best_f1": best_val_f1}
            save_json(cv_results, os.path.join(fold_dir, "cv_results.json"))
            best_epochs.append(best_epoch)


if __name__ == "__main__":
    main()

