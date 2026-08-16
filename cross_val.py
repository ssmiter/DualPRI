#!/usr/bin/env python3
'Internal utilities for the iSCALE research workflow.'
import os
import argparse
import torch
import numpy as np
import random
import json
import pickle
import pandas as pd
from datetime import datetime

from config import SHUFFLE, DEFAULT_CHUNK_SIZE, DEFAULT_DATASET_PATH
from utils import (calculate_feature_dimensions, get_feature_type_name,
                   pdb_based_kfold_split, pdb_based_kfold_split_with_randomness, load_fold_splits, create_fold_dataloaders,
                   visualize_cv_results, random_kfold_split, )
from model.utils.loader.enhanced_contact_data_loader import DEFAULT_CONTACT_THRESHOLDS, EnhancedProteinRNADataLoader
from model_factory import create_model
from trainer import train_model, SemiSupervisedProteinRNATrainer


def parse_args():
    'Parse args.'
    parser = argparse.ArgumentParser(description='PDB-based cross-validation for iSCALE')


    parser.add_argument('--data_path', type=str, default=str(DEFAULT_DATASET_PATH),
                        help='Path to the processed protein-RNA dataset (see docs/DATA.md).')
    parser.add_argument('--feature_type', type=int, default=1, choices=[0, 1, 2, 3],
                        help='Feature set: 0=base, 1=distance, 2=intensity, 3=all.')

    parser.add_argument('--esm2_cache_dir', type=str, default="./dataset_process/esm2_features",
                        help='Directory containing cached ESM2 features.')
    parser.add_argument('--check_esm2_features', action='store_true', default=True,
                        help='Check whether cached ESM2 features are available.')
    parser.add_argument('--no_check_esm2', action='store_true', default=False,
                        help='Skip the ESM2 feature availability check.')
    parser.add_argument('--force_recompute', action='store_true', default=False,
                        help='Recompute contact features instead of using the cache.')
    parser.add_argument('--cache_dir', type=str, default="./contact_cache",
                        help='Contact-feature cache directory.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size.')
    parser.add_argument('--no_reverse', default=False,
                        action='store_true', help='Do not add reverse-mutation samples.')
    parser.add_argument('--contact_thresholds', type=str, default=None,
                        help='Comma-separated contact thresholds in angstroms.')


    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of cross-validation folds.')
    parser.add_argument('--redivide', action='store_true', default=False,
                        help='Recreate fold assignments even when a split file exists.')
    parser.add_argument('--split_method', type=str, default='random')

    parser.add_argument('--model', type=str, default='iscale',
                        choices=['iscale', 'dualssd'],
                        help='Model name. "dualssd" is retained as a legacy alias for iSCALE.')
    parser.add_argument('--protein_channels', type=int, default=None,  # 41
                        help='Protein feature channels; inferred when omitted.')
    parser.add_argument('--rna_channels', type=int, default=None,  # 5
                        help='RNA feature channels; inferred when omitted.')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden channels.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of model layers.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability.')


    parser.add_argument('--d_state', type=int, default=32,
                        help='SSD state dimension.')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='SSD convolution width.')
    parser.add_argument('--expand', type=int, default=2,
                        help='SSD expansion factor.')
    parser.add_argument('--headdim', type=int, default=16,
                        help='SSD head dimension.')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='SSD chunk size; also used for contact-feature calculation.')
    parser.add_argument('--aux_weight', type=float, default=0.2,
                        help='Auxiliary-task loss weight.')


    parser.add_argument('--learning_rate', type=float, default=0.0008,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Maximum training epochs.')
    parser.add_argument('--patience', type=int, default=40,
                        help='Early-stopping patience.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Run on the CPU.')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='Checkpoint interval in epochs.')


    parser.add_argument('--output_dir', type=str, default='./cv_results',
                        help='Output directory.')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Optional experiment name.')

    return parser.parse_args()


def setup_environment(args):
    'Setup environment.'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')



    feature_type_labels = {
        0: "base", 1: "dist", 2: "int", 3: "full",
        4: "esm2", 5: "esm2_dist", 6: "esm2_int", 7: "esm2_full"
    }
    feature_label = feature_type_labels[args.feature_type]

    if args.experiment_name:
        output_dir = os.path.join(args.output_dir,
                                  f"{args.experiment_name}_{feature_label}_seed{args.seed}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.model}_{feature_label}_seed{args.seed}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    return device, output_dir


def save_config(args, output_dir):
    'Save config.'
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def check_esm2_features_availability(esm2_cache_dir, verbose=True):
    'Check esm2 features availability.'
    if not os.path.exists(esm2_cache_dir):
        if verbose:
            print(f"ESM2 feature directory not found: {esm2_cache_dir}")
        return False


    esm2_files = [f for f in os.listdir(esm2_cache_dir) if f.endswith('_esm2.pt')]

    if verbose:
        if len(esm2_files) > 0:
            print(f"Found {len(esm2_files)} ESM2 feature files.")
        else:
            print("No feature files were found in the ESM2 directory.")

    return len(esm2_files) > 0


def is_esm2_feature_type(feature_type):
    'Is esm2 feature type.'
    return feature_type in [4, 5, 6, 7]
def run_kfold_cross_validation(args, device, output_dir):
    'Run kfold cross validation.'

    if is_esm2_feature_type(args.feature_type):
        print(f"ESM2 feature set: {get_feature_type_name(args.feature_type)}")

        if not check_esm2_features_availability(args.esm2_cache_dir):
            print("ESM2 features are unavailable.")
            print("Run the ESM2 feature-extraction step first:")
            print(f"   python dataset_esm2_simplified.py")
            raise FileNotFoundError(f"ESM2 features unavailable: {args.esm2_cache_dir}")


    contact_thresholds = DEFAULT_CONTACT_THRESHOLDS
    if args.contact_thresholds:
        try:
            contact_thresholds = [float(x.strip()) for x in args.contact_thresholds.split(',')]
            print(f"Contact thresholds: {contact_thresholds}")
        except Exception as e:
            print(f"Could not parse contact thresholds ({e}); using defaults.")
            contact_thresholds = DEFAULT_CONTACT_THRESHOLDS
    else:
        print(f"Default contact thresholds: {contact_thresholds}")


    print("Loading dataset...")


    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)


    base_protein_channels = 41
    base_rna_channels = 5


    if args.protein_channels is None or args.rna_channels is None:
        protein_channels, rna_channels = calculate_feature_dimensions(
            base_protein_channels=base_protein_channels,
            base_rna_channels=base_rna_channels,
            feature_type=args.feature_type,
            contact_thresholds=contact_thresholds
        )


        if args.protein_channels is None:
            args.protein_channels = protein_channels
        if args.rna_channels is None:
            args.rna_channels = rna_channels
    else:

        protein_channels = args.protein_channels
        rna_channels = args.rna_channels

    protein_channels, rna_channels = calculate_feature_dimensions(
        base_protein_channels=base_protein_channels,
        base_rna_channels=base_rna_channels,
        feature_type=args.feature_type,
        contact_thresholds=contact_thresholds
    )


    args.protein_channels = protein_channels
    args.rna_channels = rna_channels


    print(f"Feature set: {get_feature_type_name(args.feature_type)}")
    print(f"Input channels: protein={protein_channels}, RNA={rna_channels}")
    print(f"Contact thresholds: {len(contact_thresholds)}; classes={len(contact_thresholds) + 1}")


    data_loader = EnhancedProteinRNADataLoader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_ratio=0,
        test_ratio=0,
        add_reverse=not args.no_reverse,
        seed=args.seed,
        shuffle=SHUFFLE,
        feature_type=args.feature_type,
        force_recompute=args.force_recompute,
        # cache_dir="./contact_cache",
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        contact_thresholds=contact_thresholds,
        esm2_cache_dir=args.esm2_cache_dir,
        check_esm2_features=args.check_esm2_features and not args.no_check_esm2
    )


    data_list = data_loader.data_list
    print(f"Loaded {len(data_list)} samples.")


    fold_dir = os.path.join(output_dir, 'folds')
    fold_indices_file = os.path.join(fold_dir, 'fold_indices.npz')

    if args.redivide or not os.path.exists(fold_indices_file):

        if args.split_method == 'pdb_based':
            # splits = pdb_based_kfold_split(
            splits = pdb_based_kfold_split_with_randomness(
                data_list=data_list,
                k=args.k_folds,
                seed=args.seed,
                output_dir=fold_dir,
                visualize=True
            )
        else:  # random
            splits = random_kfold_split(
                data_list=data_list,
                k=args.k_folds,
                seed=args.seed,
                output_dir=fold_dir,
                visualize=True
            )
    else:

        splits = load_fold_splits(fold_indices_file)


    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print(f"\n===== Training fold {fold_idx + 1}/{args.k_folds} =====")


        train_loader, val_loader = create_fold_dataloaders(
            data_list=data_list,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=args.batch_size
        )


        model = create_model(
            args.model,
            protein_channels=args.protein_channels,
            rna_channels=args.rna_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,

            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            headdim=args.headdim,
            chunk_size=args.chunk_size,
            aux_weight=args.aux_weight
        )


        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")


        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=fold_output_dir,
            max_epochs=args.epochs,
            patience=args.patience,
            checkpoint_interval=args.checkpoint_interval
        )


        best_val_metrics = trainer.val_metrics[trainer.best_epoch]


        all_preds = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                wild_data, mutant_data, rna_data, ddg = batch
                wild_data = wild_data.to(device)
                mutant_data = mutant_data.to(device)
                rna_data = rna_data.to(device)

                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=device)

                output = model(wild_data, mutant_data, rna_data)


                if isinstance(output, dict):
                    pred = output['ddg']
                elif isinstance(output, torch.Tensor):
                    pred = output
                elif isinstance(output, tuple):
                    pred = output[0]

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())


        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        np.save(os.path.join(fold_output_dir, "predictions.npy"),
                {'predictions': all_preds, 'true_values': all_targets})


        fold_result = {
            'fold': fold_idx + 1,
            'best_epoch': trainer.best_epoch + 1,
            'mse': best_val_metrics['mse'],
            'mae': best_val_metrics['mae'],
            'pcc': best_val_metrics['pcc']
        }
        fold_results.append(fold_result)

        print(f"Fold {fold_idx + 1} best result (epoch {fold_result['best_epoch']}):")
        print(f"  MSE: {fold_result['mse']:.4f}")
        print(f"  MAE: {fold_result['mae']:.4f}")
        print(f"  PCC: {fold_result['pcc']:.4f}")


    avg_metrics = {
        'mse': np.mean([r['mse'] for r in fold_results]),
        'mse_std': np.std([r['mse'] for r in fold_results]),
        'mae': np.mean([r['mae'] for r in fold_results]),
        'mae_std': np.std([r['mae'] for r in fold_results]),
        'pcc': np.mean([r['pcc'] for r in fold_results]),
        'pcc_std': np.std([r['pcc'] for r in fold_results])
    }


    with open(os.path.join(output_dir, 'cv_results_summary.txt'), 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Feature set: {args.feature_type} ({get_feature_type_name(args.feature_type)})\n")
        f.write(f"K-fold cross-validation (k={args.k_folds})\n")
        f.write("======================\n\n")

        for fold_result in fold_results:
            f.write(f"Fold {fold_result['fold']}:\n")
            f.write(f"  Best epoch: {fold_result['best_epoch']}\n")
            f.write(f"  MSE: {fold_result['mse']:.4f}\n")
            f.write(f"  MAE: {fold_result['mae']:.4f}\n")
            f.write(f"  PCC: {fold_result['pcc']:.4f}\n\n")

        f.write("Mean performance:\n")
        f.write(f"  MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}\n")
        f.write(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}\n")
        f.write(f"  PCC: {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}\n")


    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)


    visualize_cv_results(fold_results, avg_metrics, output_dir)


    print("\n===== Cross-validation summary =====")
    print(f"Model: {args.model}")
    print(f"Feature set: {args.feature_type} ({get_feature_type_name(args.feature_type)})")
    print("Mean performance:")
    print(f"  MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
    print(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}")
    print(f"  PCC: {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}")

    return avg_metrics



def main():
    args = parse_args()


    device, output_dir = setup_environment(args)


    config = vars(args).copy()
    config.update({
        'device': str(device),
        'feature_type_name': get_feature_type_name(args.feature_type),
        'is_esm2_feature': is_esm2_feature_type(args.feature_type)
    })


    save_config(args, output_dir)

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    avg_metrics = run_kfold_cross_validation(args, device, output_dir)

    print(f"Cross-validation results saved to {output_dir}")
    print(f"Final mean PCC: {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}")

    return avg_metrics


if __name__ == "__main__":
    main()
