from __future__ import annotations
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
import torch
import optuna
from optuna.trial import TrialState
from torch.utils.data import ConcatDataset, DataLoader
import pandas as pd

# Optuna hyperparameter optimization framework adapted from 
# https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py
# Thank you for publishing!

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

from src.utils import get_output_dir, set_determinism, read_hdf5, log, save_test, save_json, save_cm, save_cm_norm, save_labelmap, save_unknown_IDs, save_optuna_results, save_cm_norm_raw, save_indiv_results
from src.classify.data import encode_category, add_features, ProteinDataset, GenomicContextDataset, analyze_encoder, build_dataloaders, collate_fn, filter_unknowns, filter_cluster_representatives, filter_nan_embeddings
from src.classify.model import MLP, ContextTransformer, CNN_MLP, CNN, CNN_Flattened
from src.classify.train import fit, objective_mlp, objective_cnn_mlp, objective_cnn, objective_cnn_flattened

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

    train_loader, val_loader, test_loader = build_dataloaders(
        ds,
        batch_size=cfg.optim.batch_size,
        val_fraction=cfg.data.val_size,
        test_fraction=cfg.data.test_size,
        seed=cfg.seed,
        context=cfg.context.bool,
        num_workers=cfg.data.num_workers,
        optimize=cfg.optimize.bool
    )

    # include bacterial and viral proteins
    if cfg.bac_vir.bool:
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

        metadata_combined = pd.concat([metadata_knowns_encoded, metadata_bac_vir], ignore_index=True)
        # get number of classes and calculate label weights
        num_categories, label_map, label_weights = analyze_encoder(LE_SC, metadata_combined, colname)
    
    else:
        # get number of classes and calculate label weights
        num_categories, label_map, label_weights = analyze_encoder(LE_SC, metadata_knowns_encoded, colname)

    # Loss function with class weights to handle imbalance
    logger.info(f"label_weights.values: {label_weights.values}")
    weights = torch.tensor(label_weights.values, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    in_channels = cfg.data.embed_dim
    num_classes = num_categories

    if cfg.optimize.bool:
        # Pre‑training diagnostics
        logger.info(
            "Dataset: total=%d | train=%d | test=%d",
            len(ds),
            len(train_loader.dataset),  # type: ignore[arg-type]
            len(test_loader.dataset),  # type: ignore[arg-type]
        )

        epochs = cfg.optimize.epochs

        batch_size_train = cfg.optimize.batch_size_train
        batch_size_test = cfg.optimize.batch_size_test

        num_train_examples = 200 * batch_size_train
        num_test_examples = 5 * batch_size_test

        logger.info(
            "Number of examples: train=%d | test=%d",
            num_train_examples, 
            num_test_examples)

        # Create an Optuna study to maximize test F1
        study = optuna.create_study(direction="maximize")

        # Set up logging
        log_fn = log(logger)

        if cfg.data.pooled:
            if cfg.context.bool:
                raise NotImplementedError

            else:
                study.optimize(lambda trial: objective_mlp(trial, train_loader, test_loader, loss_fn, epochs, device, in_channels, num_classes, 0, batch_size_train, num_train_examples, batch_size_test, num_test_examples, log_fn, cfg.train.log_interval), n_trials=cfg.optimize.n_trials)

        else:
            if cfg.optimize.cnn_mlp:
                study.optimize(lambda trial: objective_cnn_mlp(trial, train_loader, test_loader, loss_fn, epochs, device, in_channels, num_classes, 0, cfg.optimize.cnn_use_linear_attention, cfg.optimize.cnn_use_non_linear_attention, batch_size_train, num_train_examples, batch_size_test, num_test_examples, log_fn, cfg.train.log_interval), n_trials=cfg.optimize.n_trials)
            elif cfg.optimize.cnn_flattened:
                study.optimize(lambda trial: objective_cnn_flattened(trial, train_loader, test_loader, loss_fn, epochs, device, in_channels, num_classes, cfg.optimize.mean_max, 0, batch_size_train, num_train_examples, batch_size_test, num_test_examples, log_fn, cfg.train.log_interval), n_trials=cfg.optimize.n_trials)
            else: 
                study.optimize(lambda trial: objective_cnn(trial, train_loader, test_loader, loss_fn, epochs, device, in_channels, num_classes, cfg.optimize.mean_max, 0, cfg.optimize.cnn_use_linear_attention, cfg.optimize.cnn_use_non_linear_attention, batch_size_train, num_train_examples, batch_size_test, num_test_examples, log_fn, cfg.train.log_interval), n_trials=cfg.optimize.n_trials)

        # Results
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        # Display the study statistics
        logger.info("\nStudy statistics: ")
        logger.info("  Number of finished trials: ", len(study.trials))
        logger.info("  Number of pruned trials: ", len(pruned_trials))
        logger.info("  Number of complete trials: ", len(complete_trials))

        trial = study.best_trial
        logger.info("Best trial:")
        logger.info("  Value: ", trial.value)
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))

        # Save results to csv file
        df_optuna = save_optuna_results(study, os.path.join(output_dir, f"optuna_results.csv"))

        # Display results in a dataframe
        logger.info("\nOverall Results (ordered by F1):\n {}".format(df_optuna))

        # Find the most important hyperparameters
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)

        # Display the most important hyperparameters
        logger.info('\nMost important hyperparameters:')
        for key, value in most_important_parameters.items():
            logger.info('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
                

    # if config parameters should be used, not optuna
    else:
        # Pre‑training diagnostics
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
                        n_feats=0 if cfg.features.bool != True else cfg.features.top_n,
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
                        n_feats=0 if cfg.features.bool != True else cfg.features.top_n,
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
            os.makedirs(os.path.join(output_dir, "predictions", model_key), exist_ok=True)

            results = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=cfg.optim.epochs,
                device=device,
                log_fn=log_fn,
                log_interval=cfg.train.log_interval,
                encoder=LE_SC,
                path=os.path.join(output_dir, "predictions", model_key)
            )

            # report for best validation f1 state
            bv_confusion_matrix = results.pop("bv_test_confusion_matrix")
            bv_confusion_matrix_norm = results.pop("bv_test_confusion_matrix_norm")
            bv_test_targets = results.pop("bv_test_targets")
            bv_test_preds = results.pop("bv_test_preds")
            bv_test_ids = results.pop("bv_test_ids")
            bv_test_probs = results.pop("bv_test_probs")

            keys = ["best_val_f1_weighted", "bv_test_loss", "bv_test_acc", "bv_test_macro_f1", "bv_test_weighted_f1", "bv_test_macro_recall", "bv_test_weighted_recall", "bv_test_macro_precision", "bv_test_weighted_precision"]
            bv_results = {k: results.get(k) for k in keys}
            logger.info("Final results (best val f1 state): " + ", ".join(f"{k}={v:.4f}" for k, v in bv_results.items()))

            indiv_keys = ["bv_test_indiv_f1", "bv_test_indiv_recall", "bv_test_indiv_precision", "bv_test_support"]
            bv_indiv_results = {k: results.get(k) for k in indiv_keys}

            save_test(bv_test_targets, bv_test_preds, bv_test_probs, bv_test_ids, LE_SC, os.path.join(output_dir, "bv_test.tsv"))

            bv_state = results.pop("bv_best_state")
            model.load_state_dict(bv_state)

            model_path = os.path.join(output_dir, f"bv_{model_key}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "results": bv_results,
                    "label_map": label_map,
                },
                model_path,
            )
            logger.info(f"Saved model checkpoint to {model_path}")

            save_json(bv_results, os.path.join(output_dir, f"bv_results_{model_key}.json"))
            save_indiv_results(bv_indiv_results, os.path.join(output_dir, f"bv_indiv_results_{model_key}.csv"))
            save_cm(bv_confusion_matrix, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"bv_cm_{model_key}.png"))
            save_cm_norm(bv_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"bv_cm_{model_key}_norm.png"))
            save_cm_norm_raw(bv_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"bv_cm_{model_key}_values.csv"))

            # report for early stopping state
            es_confusion_matrix = results.pop("es_test_confusion_matrix")
            es_confusion_matrix_norm = results.pop("es_test_confusion_matrix_norm")
            es_test_targets = results.pop("es_test_targets")
            es_test_preds = results.pop("es_test_preds")
            es_test_ids = results.pop("es_test_ids")
            es_test_probs = results.pop("es_test_probs")

            keys = ["earliest_lowest_val_loss", "es_test_loss", "es_test_acc", "es_test_macro_f1", "es_test_weighted_f1", "es_test_macro_recall", "es_test_weighted_recall", "es_test_macro_precision", "es_test_weighted_precision"]
            es_results = {k: results.get(k) for k in keys}
            logger.info("Final results (early stopping state): " + ", ".join(f"{k}={v:.4f}" for k, v in es_results.items()))

            indiv_keys = ["es_test_indiv_f1", "es_test_indiv_recall", "es_test_indiv_precision", "es_test_support"]
            es_indiv_results = {k: results.get(k) for k in indiv_keys}

            save_test(es_test_targets, es_test_preds, es_test_probs, es_test_ids, LE_SC, os.path.join(output_dir, "es_test.tsv"))

            es_state = results.pop("es_state")
            model.load_state_dict(es_state)

            model_path = os.path.join(output_dir, f"es_{model_key}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "results": es_results,
                    "label_map": label_map,
                },
                model_path,
            )
            logger.info(f"Saved model checkpoint to {model_path}")

            save_json(es_results, os.path.join(output_dir, f"es_results_{model_key}.json"))
            save_indiv_results(es_indiv_results, os.path.join(output_dir, f"es_indiv_results_{model_key}.csv"))
            save_cm(es_confusion_matrix, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"es_cm_{model_key}.png"))
            save_cm_norm(es_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"es_cm_{model_key}_norm.png"))
            save_cm_norm_raw(es_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"es_cm_{model_key}_values.csv"))


            # report for epoch30 state
            e30_confusion_matrix = results.pop("e30_test_confusion_matrix")
            e30_confusion_matrix_norm = results.pop("e30_test_confusion_matrix_norm")
            e30_test_targets = results.pop("e30_test_targets")
            e30_test_preds = results.pop("e30_test_preds")
            e30_test_ids = results.pop("e30_test_ids")
            e30_test_probs = results.pop("e30_test_probs")

            keys = ["earliest_lowest_val_loss", "e30_test_loss", "e30_test_acc", "e30_test_macro_f1", "e30_test_weighted_f1", "e30_test_macro_recall", "e30_test_weighted_recall", "e30_test_macro_precision", "e30_test_weighted_precision"]
            e30_results = {k: results.get(k) for k in keys}
            logger.info("Final results (epoch 30 state): " + ", ".join(f"{k}={v:.4f}" for k, v in e30_results.items()))

            indiv_keys = ["e30_test_indiv_f1", "e30_test_indiv_recall", "e30_test_indiv_precision", "e30_test_support"]
            e30_indiv_results = {k: results.get(k) for k in indiv_keys}

            save_test(e30_test_targets, e30_test_preds, e30_test_probs, e30_test_ids, LE_SC, os.path.join(output_dir, "e30_test.tsv"))

            e30_state = results.pop("e30_state")
            model.load_state_dict(e30_state)

            model_path = os.path.join(output_dir, f"e30_{model_key}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "results": e30_results,
                    "label_map": label_map,
                },
                model_path,
            )
            logger.info(f"Saved model checkpoint to {model_path}")

            save_json(e30_results, os.path.join(output_dir, f"e30_results_{model_key}.json"))
            save_indiv_results(e30_indiv_results, os.path.join(output_dir, f"e30_indiv_results_{model_key}.csv"))
            save_cm(e30_confusion_matrix, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"e30_cm_{model_key}.png"))
            save_cm_norm(e30_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"e30_cm_{model_key}_norm.png"))
            save_cm_norm_raw(e30_confusion_matrix_norm, LE_SC.transform(LE_SC.classes_), os.path.join(output_dir, f"e30_cm_{model_key}_values.csv"))


            # save general info
            log_objects = results.pop("log_objects")
            normal_dict_log_objects = {k: dict(v) for k, v in log_objects.items()}
            save_json(normal_dict_log_objects, os.path.join(output_dir, f"log_objects_{model_key}.json"))

            save_labelmap(label_map, os.path.join(output_dir, f"label_map.txt"))
            save_unknown_IDs(metadata_unknowns, os.path.join(output_dir, f"protein_IDs_unknown.csv"))

            logger.info(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

