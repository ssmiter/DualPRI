
import json
import datetime
import pandas as pd
import seaborn as sns
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from config import DEFAULT_CONTACT_THRESHOLDS
from torch_geometric.loader import DataLoader
from scipy.stats import gaussian_kde
from model.utils.loader.enhanced_contact_data_loader import collate_protein_rna_triple
from config import NUM_WORKERS, PREFETCH_FACTOR
import matplotlib
import matplotlib.patheffects as path_effects
matplotlib.use('Agg')  # Set non-interactive backend





def calculate_feature_dimensions(base_protein_channels, base_rna_channels, feature_type, contact_thresholds=None):
    'Calculate feature dimensions.'
    if contact_thresholds is None:
        contact_thresholds = DEFAULT_CONTACT_THRESHOLDS


    esm2_dim = 1280


    contact_dist_dim = len(contact_thresholds) + 1
    contact_int_dim = 1


    if feature_type == 0:

        protein_channels = base_protein_channels
        rna_channels = base_rna_channels

    elif feature_type == 1:

        protein_channels = base_protein_channels + contact_dist_dim
        rna_channels = base_rna_channels + contact_dist_dim

    elif feature_type == 2:

        protein_channels = base_protein_channels + contact_int_dim
        rna_channels = base_rna_channels + contact_int_dim

    elif feature_type == 3:

        protein_channels = base_protein_channels + contact_dist_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_dist_dim + contact_int_dim

    elif feature_type == 4:

        protein_channels = esm2_dim
        rna_channels = base_rna_channels

    elif feature_type == 5:

        protein_channels = esm2_dim + contact_dist_dim
        rna_channels = base_rna_channels + contact_dist_dim

    elif feature_type == 6:

        protein_channels = esm2_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_int_dim

    elif feature_type == 7:

        protein_channels = esm2_dim + contact_dist_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_dist_dim + contact_int_dim

    else:

        print(f"Warning: unknown feature type {feature_type}; using base features.")
        protein_channels = base_protein_channels
        rna_channels = base_rna_channels

    return protein_channels, rna_channels



def get_feature_type_name(feature_type):
    'Get feature type name.'
    feature_type_names = {

        0: "base features",
        1: "distance-distribution features",
        2: "contact-intensity features",
        3: "all multiscale features",


        4: "ESM2 features",
        5: "ESM2 and distance-distribution features",
        6: "ESM2 and contact-intensity features",
        7: "ESM2 and all multiscale features"
    }
    return feature_type_names.get(feature_type, f"unknown feature type ({feature_type})")



def is_esm2_feature_type(feature_type):
    'Is esm2 feature type.'
    return feature_type in [4, 5, 6, 7]


def get_feature_type_details(feature_type, contact_thresholds=None):
    'Get feature type details.'
    if contact_thresholds is None:
        contact_thresholds = DEFAULT_CONTACT_THRESHOLDS


    base_protein_dim = 41
    base_rna_dim = 5
    esm2_dim = 1280
    contact_dist_dim = len(contact_thresholds) + 1
    contact_int_dim = 1


    protein_dim, rna_dim = calculate_feature_dimensions(
        base_protein_dim, base_rna_dim, feature_type, contact_thresholds
    )


    components = {
        'uses_base_protein': feature_type in [0, 1, 2, 3],
        'uses_esm2': feature_type in [4, 5, 6, 7],
        'uses_contact_dist': feature_type in [1, 3, 5, 7],
        'uses_contact_int': feature_type in [2, 3, 6, 7],
        'uses_any_contact': feature_type in [1, 2, 3, 5, 6, 7]
    }

    return {
        'feature_type': feature_type,
        'name': get_feature_type_name(feature_type),
        'protein_dim': protein_dim,
        'rna_dim': rna_dim,
        'components': components,
        'dimensions': {
            'base_protein': base_protein_dim if components['uses_base_protein'] else 0,
            'esm2': esm2_dim if components['uses_esm2'] else 0,
            'contact_dist': contact_dist_dim if components['uses_contact_dist'] else 0,
            'contact_int': contact_int_dim if components['uses_contact_int'] else 0
        }
    }

def get_model_specific_params(model_name):
    'Get model specific params.'
    model_params = {

        'iscale': {
            'batch_size': 16,
            'learning_rate': 0.0008,
        },
        'dualssd': {
            'batch_size': 16,
            'learning_rate': 0.0008,
        },
        'transformer': {
            'batch_size': 8,
            'learning_rate': 0.0005,
        },
        'graph_transformer': {
            'batch_size': 8,
            'learning_rate': 0.0005,
        },
        'gcn': {
            'batch_size': 16,
            'learning_rate': 0.0008,
        },
        'gat': {
            'batch_size': 16,
            'learning_rate': 0.0008,
        },
    }


    return model_params.get(model_name, {})



def random_kfold_split(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    'Random kfold split.'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating random {k}-fold cross-validation split")
    print(f"Total samples: {len(data_list)}")


    indices = list(range(len(data_list)))
    random.shuffle(indices)


    fold_size = len(indices) // k
    remainder = len(indices) % k


    fold_indices = []
    start = 0
    for i in range(k):

        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        fold_indices.append(indices[start:end])
        start = end


    splits = []
    fold_sample_counts = [len(fold) for fold in fold_indices]

    for i in range(k):
        val_indices = fold_indices[i]
        train_indices = [idx for j, fold in enumerate(fold_indices) if j != i for idx in fold]
        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")


    fold_pdb_info = []
    for i, indices in enumerate(fold_indices):
        pdb_counts = {}
        for idx in indices:
            wild_data = data_list[idx][0]
            pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
            pdb_counts[pdb_id] = pdb_counts.get(pdb_id, 0) + 1

        fold_pdb_info.append({
            'fold_id': i,
            'pdb_count': len(pdb_counts),
            'sample_count': len(indices),
            'percentage': len(indices) / len(data_list) * 100,
            'pdbs': list(pdb_counts.keys()),
            'pdb_distribution': pdb_counts
        })


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


        with open(os.path.join(output_dir, 'fold_info.txt'), "w") as f:
            f.write(f"Random {k}-fold Cross Validation\n")
            f.write(f"===========================\n")
            f.write(f"Total samples: {len(data_list)}\n")
            f.write(f"Random seed: {seed}\n\n")

            for i, info in enumerate(fold_pdb_info):
                f.write(f"Fold {i + 1}:\n")
                f.write(f"  PDB count: {info['pdb_count']}\n")
                f.write(f"  Sample count: {info['sample_count']} ({info['percentage']:.1f}%)\n")
                f.write(f"  PDB IDs: {', '.join(info['pdbs'])}\n\n")


                f.write(f"  PDB distribution in this fold:\n")
                for pdb, count in info['pdb_distribution'].items():
                    f.write(f"    {pdb}: {count} samples\n")
                f.write("\n")


        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )


    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'Random {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)


        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')


        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits


def pdb_based_kfold_split_with_randomness(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    'Pdb based kfold split with randomness.'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating PDB-based {k}-fold cross-validation split with enhanced randomness")
    print(f"Total samples: {len(data_list)}")


    pdb_groups = {}
    for idx, (wild_data, mutant_data, rna_data, ddg) in enumerate(data_list):
        pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(idx)


    pdb_ids = list(pdb_groups.keys())


    random.shuffle(pdb_ids)

    pdb_sample_counts = {pdb: len(indices) for pdb, indices in pdb_groups.items()}

    print(f"Dataset contains {len(pdb_ids)} different PDB structures")



    size_groups = {}
    for pdb in pdb_ids:
        count = pdb_sample_counts[pdb]
        if count not in size_groups:
            size_groups[count] = []
        size_groups[count].append(pdb)


    pdb_by_size = []
    for count in sorted(size_groups.keys(), reverse=True):
        pdbs = size_groups[count]
        random.shuffle(pdbs)
        for pdb in pdbs:
            pdb_by_size.append((pdb, count))


    folds = [[] for _ in range(k)]
    fold_sample_counts = [0] * k


    for pdb, count in pdb_by_size:

        min_fold_idx = fold_sample_counts.index(min(fold_sample_counts))
        folds[min_fold_idx].append(pdb)
        fold_sample_counts[min_fold_idx] += count


    fold_details = []
    for i, fold_pdbs in enumerate(folds):
        fold_info = {
            "fold_id": i,
            "pdb_count": len(fold_pdbs),
            "sample_count": fold_sample_counts[i],
            "percentage": fold_sample_counts[i] / len(data_list) * 100,
            "pdbs": fold_pdbs
        }
        fold_details.append(fold_info)

        print(f"Fold {i + 1}: {len(fold_pdbs)} PDBs, {fold_sample_counts[i]} samples ({fold_info['percentage']:.1f}%)")


    splits = []
    for i in range(k):
        val_indices = []
        for pdb in folds[i]:
            val_indices.extend(pdb_groups[pdb])

        train_indices = []
        for j in range(k):
            if j != i:
                for pdb in folds[j]:
                    train_indices.extend(pdb_groups[pdb])


        random.shuffle(val_indices)
        random.shuffle(train_indices)

        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


        with open(os.path.join(output_dir, 'fold_info.txt'), "w") as f:
            f.write(f"PDB-based {k}-fold Cross Validation (with randomness)\n")
            f.write(f"===========================\n")
            f.write(f"Total samples: {len(data_list)}\n")
            f.write(f"Total PDB structures: {len(pdb_ids)}\n")
            f.write(f"Random seed: {seed}\n\n")

            for i, info in enumerate(fold_details):
                f.write(f"Fold {i + 1}:\n")
                f.write(f"  PDB count: {info['pdb_count']}\n")
                f.write(f"  Sample count: {info['sample_count']} ({info['percentage']:.1f}%)\n")
                f.write(f"  PDB IDs: {', '.join(info['pdbs'])}\n\n")


        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )


    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'PDB-based {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)


        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')


        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits

def pdb_based_kfold_split(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    'Pdb based kfold split.'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating PDB-based {k}-fold cross-validation split")
    print(f"Total samples: {len(data_list)}")


    pdb_groups = {}
    for idx, (wild_data, mutant_data, rna_data, ddg) in enumerate(data_list):
        pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(idx)


    pdb_ids = list(pdb_groups.keys())
    pdb_sample_counts = {pdb: len(indices) for pdb, indices in pdb_groups.items()}

    print(f"Dataset contains {len(pdb_ids)} different PDB structures")


    pdb_by_size = sorted(pdb_sample_counts.items(), key=lambda x: x[1], reverse=True)


    folds = [[] for _ in range(k)]
    fold_sample_counts = [0] * k


    for pdb, count in pdb_by_size:

        min_fold_idx = fold_sample_counts.index(min(fold_sample_counts))
        folds[min_fold_idx].append(pdb)
        fold_sample_counts[min_fold_idx] += count


    fold_details = []
    for i, fold_pdbs in enumerate(folds):
        fold_info = {
            "fold_id": i,
            "pdb_count": len(fold_pdbs),
            "sample_count": fold_sample_counts[i],
            "percentage": fold_sample_counts[i] / len(data_list) * 100,
            "pdbs": fold_pdbs
        }
        fold_details.append(fold_info)

        print(f"Fold {i + 1}: {len(fold_pdbs)} PDBs, {fold_sample_counts[i]} samples ({fold_info['percentage']:.1f}%)")


    splits = []
    for i in range(k):
        val_indices = []
        for pdb in folds[i]:
            val_indices.extend(pdb_groups[pdb])

        train_indices = []
        for j in range(k):
            if j != i:
                for pdb in folds[j]:
                    train_indices.extend(pdb_groups[pdb])

        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")


    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


        with open(os.path.join(output_dir, 'fold_info.txt'), "w") as f:
            f.write(f"PDB-based {k}-fold Cross Validation\n")
            f.write(f"===========================\n")
            f.write(f"Total samples: {len(data_list)}\n")
            f.write(f"Total PDB structures: {len(pdb_ids)}\n")
            f.write(f"Random seed: {seed}\n\n")

            for i, info in enumerate(fold_details):
                f.write(f"Fold {i + 1}:\n")
                f.write(f"  PDB count: {info['pdb_count']}\n")
                f.write(f"  Sample count: {info['sample_count']} ({info['percentage']:.1f}%)\n")
                f.write(f"  PDB IDs: {', '.join(info['pdbs'])}\n\n")


        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )


    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'PDB-based {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)


        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')


        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits


def load_fold_splits(npz_path):
    'Load fold splits.'
    data = np.load(npz_path)
    k = len([key for key in data.keys() if key.startswith('train_fold')])

    splits = []
    for i in range(k):
        train_indices = data[f'train_fold_{i + 1}']
        val_indices = data[f'val_fold_{i + 1}']
        splits.append((train_indices, val_indices))

    print(f"Loaded {k} fold assignments from {npz_path}")
    return splits


def create_fold_dataloaders(data_list, train_indices, val_indices, batch_size=16, num_workers=NUM_WORKERS,
                            prefetch_factor=PREFETCH_FACTOR):
    'Create fold dataloaders.'

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]


    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_protein_rna_triple,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_protein_rna_triple,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    return train_loader, val_loader


def visualize_cv_results(fold_results, avg_metrics, output_dir):
    'Visualize cv results.'

    plt.figure(figsize=(12, 6))


    folds = [r['fold'] for r in fold_results]
    pccs = [r['pcc'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    mses = [r['mse'] for r in fold_results]


    ax1 = plt.subplot(1, 3, 1)
    bars = ax1.bar(folds, pccs, color='skyblue')
    ax1.axhline(y=avg_metrics['pcc'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["pcc"]:.4f} ± {avg_metrics["pcc_std"]:.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('PCC')
    ax1.set_title('PCC by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{pccs[i]:.4f}', ha='center', va='bottom')


    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.bar(folds, maes, color='lightgreen')
    ax2.axhline(y=avg_metrics['mae'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["mae"]:.4f} ± {avg_metrics["mae_std"]:.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{maes[i]:.4f}', ha='center', va='bottom')


    ax3 = plt.subplot(1, 3, 3)
    bars = ax3.bar(folds, mses, color='salmon')
    ax3.axhline(y=avg_metrics['mse'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["mse"]:.4f} ± {avg_metrics["mse_std"]:.4f}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE by Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)


    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{mses[i]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_results.png'), dpi=300)
    plt.close()



    all_true = []
    all_pred = []



    for fold_idx, fold_result in enumerate(fold_results):

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        predictions_file = os.path.join(fold_dir, "predictions.npy")


        if os.path.exists(predictions_file):
            fold_data = np.load(predictions_file, allow_pickle=True).item()
            all_true.extend(fold_data['true_values'])
            all_pred.extend(fold_data['predictions'])


    if all_true and all_pred:
        metrics = {
            'PCC': avg_metrics['pcc'],
            'RMSE': np.sqrt(avg_metrics['mse'])
        }

        plot_results_enhanced_v2(
            np.array(all_true),
            np.array(all_pred),
            save_path=os.path.join(output_dir, 'prediction_scatter.png'),
            metrics=metrics,
            dataset=f"{len(fold_results)}-Fold CV",
            theme_color='blue'
        )


def plot_enhanced_histogram_v2(ax, data, bins, plot_range, vertical=True, theme_color='red'):
    """Enhanced histogram plotting with layered colors

    Args:
        ax: matplotlib axis
        data: data to plot
        bins: histogram bins
        plot_range: plot range [min, max]
        vertical: if True, plot vertical histogram, else horizontal
        theme_color: theme color for the plot
    """

    kde_points = np.linspace(plot_range[0], plot_range[1], 200)
    kde = gaussian_kde(data)
    kde_values = kde(kde_points)


    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


    dark_color = plt.get_cmap('Dark2')(plt.matplotlib.colors.to_rgba(theme_color)[0])

    if vertical:
        ax.fill_between(kde_points, kde_values, alpha=0.35, color=theme_color, zorder=1)
        ax.bar(bin_centers, hist, width=np.diff(bin_edges), alpha=0.35,
               color=theme_color, edgecolor=dark_color, linewidth=0.5, zorder=2)
        ax.plot(kde_points, kde_values, alpha=0.35, color=dark_color, linewidth=0.5, zorder=3)
    else:
        ax.fill_betweenx(kde_points, kde_values, alpha=0.35, color=theme_color, zorder=1)
        ax.barh(bin_centers, hist, height=np.diff(bin_edges), alpha=0.35,
                color=theme_color, edgecolor=dark_color, linewidth=0.5, zorder=2)
        ax.plot(kde_values, kde_points, alpha=0.35, color=dark_color, linewidth=0.5, zorder=3)


def plot_results_enhanced_v2(true_ddg, predictions, save_path, metrics=None, dataset=None, title="", theme_color='red'):
    'Plot results enhanced v2.'
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.linewidth': 2,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.weight': 'bold',
    })


    fig = plt.figure(figsize=(9, 9))

    main_ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    top_ax = fig.add_axes((0.15, 0.85, 0.7, 0.15))
    right_ax = fig.add_axes((0.85, 0.15, 0.15, 0.7))

    plot_range = [-10, 10]


    if metrics:
        pcc = metrics.get('PCC', 0)
        rmse = metrics.get('RMSE', 0)
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))


    slope, intercept = np.polyfit(true_ddg, predictions, 1)
    # data_range_min = np.percentile(true_ddg, 0.2)
    # data_range_max = np.percentile(true_ddg, 99.8)
    # fit_x = np.array([data_range_min, data_range_max])
    fit_x = np.array([-10, 10])
    fit_line = slope * fit_x + intercept


    main_ax.set_xlim(plot_range)
    main_ax.set_ylim(plot_range)
    top_ax.set_xlim(plot_range)
    right_ax.set_ylim(plot_range)

    major_ticks = np.arange(-10, 12, 2)
    minor_ticks = np.arange(-10, 11, 1)

    main_ax.set_xticks(major_ticks)
    main_ax.set_yticks(major_ticks)
    main_ax.set_xticks(minor_ticks, minor=True)
    main_ax.set_yticks(minor_ticks, minor=True)

    main_ax.tick_params(axis='both', which='major', length=6, width=2, direction='out')
    main_ax.tick_params(axis='both', which='minor', length=4, width=1.5, direction='out')

    main_ax.grid(True, which='major', linestyle='--', alpha=0.5, color='gray', zorder=1)
    main_ax.grid(True, which='minor', linestyle='--', alpha=0.2, color='gray', zorder=1)


    main_ax.scatter(true_ddg, predictions, alpha=0.6, color=theme_color, s=50, zorder=3)
    # main_ax.plot(plot_range, plot_range, '--', color='black', alpha=0.8, linewidth=1.5, zorder=2)
    main_ax.plot(fit_x, fit_line, '-', color=theme_color, alpha=0.8, linewidth=2.0, zorder=2)


    bins = np.arange(-10, 11, 1)
    plot_enhanced_histogram_v2(top_ax, true_ddg, bins, plot_range, vertical=True, theme_color=theme_color)
    plot_enhanced_histogram_v2(right_ax, predictions, bins, plot_range, vertical=False, theme_color=theme_color)


    top_ax.set_xticks([])
    top_ax.set_yticks([])
    right_ax.set_xticks([])
    right_ax.set_yticks([])


    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    top_ax.spines['top'].set_visible(False)
    top_ax.spines['right'].set_visible(False)
    top_ax.spines['left'].set_visible(False)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['bottom'].set_visible(False)


    dataset_name = f"{dataset.upper()}" if dataset else ""
    dataset_text = f"{dataset_name}\n"
    metrics_text = f"RMSE: {rmse:.3f}\nPCC   : {pcc:.3f}"

    intercept_str = f"- {abs(intercept):.3f}" if intercept < 0 else f"+ {intercept:.3f}"
    equation_text = f"y = {slope:.3f}x {intercept_str}"


    main_ax.text(0.02, 0.98, dataset_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=30,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.91, metrics_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)

    main_ax.text(0.02, 0.81, equation_text,
                 transform=main_ax.transAxes,
                 verticalalignment='top',
                 horizontalalignment='left',
                 fontsize=18,
                 fontweight='bold',
                 fontstyle='italic',
                 family='DejaVu Serif',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0,
                           edgecolor='none'),
                 zorder=5)


    main_ax.set_xlabel('True', fontsize=20, fontweight='bold')
    main_ax.set_ylabel('Prediction', fontsize=20, fontweight='bold')


    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()




def extract_metrics_from_directory(directory):
    'Extract metrics from directory.'
    try:

        metrics_path = os.path.join(directory, "best_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                return metrics.get('test', {})


        test_results_path = os.path.join(directory, "test_results.txt")
        if os.path.exists(test_results_path):
            metrics = {}
            with open(test_results_path, 'r') as f:
                for line in f:
                    if "MSE:" in line:
                        metrics['mse'] = float(line.split(":")[1].strip())
                    elif "MAE:" in line:
                        metrics['mae'] = float(line.split(":")[1].strip())
                    elif "PCC:" in line:
                        metrics['pcc'] = float(line.split(":")[1].strip())
            return metrics
    except Exception as e:
        print(f"Could not extract metrics: {e}")

    return None


def visualize_training_ratio_results(results, output_dir):
    """
    Visualize results as training ratio curve

    Parameters:
        results: List of result dictionaries
        output_dir: Output directory for plots and CSV files

    Returns:
        grouped: DataFrame with grouped statistics
    """
    if not results:
        print("No results to visualize")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Create timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw results to CSV
    raw_csv_path = os.path.join(output_dir, f"training_ratio_raw_results_{timestamp}.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"Raw results saved to: {raw_csv_path}")

    # Group by model, feature_type, and train_ratio to get statistics
    grouped = df.groupby(['model', 'feature_name', 'train_ratio']).agg({
        'pcc': ['mean', 'std', 'count'],
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std']
    }).reset_index()

    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

    # Save summary results to CSV
    summary_csv_path = os.path.join(output_dir, f"training_ratio_summary_{timestamp}.csv")
    grouped.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to: {summary_csv_path}")

    # Create plot directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot PCC vs. train_ratio for each model and feature type
    plt.figure(figsize=(12, 8))

    # Get unique models and feature types
    models = df['model'].unique()
    feature_types = df['feature_name'].unique()

    # Define a color map and marker styles
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'd', 'x', '*', 'p', 'h', 'v', '>', '<']

    # Plot with different colors/markers for each model
    for i, model in enumerate(models):
        for j, feature_type in enumerate(feature_types):
            # Filter data for this model and feature type
            model_data = grouped[
                (grouped['model'] == model) &
                (grouped['feature_name'] == feature_type)
                ]

            if model_data.empty:
                continue

            # Sort by train_ratio for proper line plotting
            model_data = model_data.sort_values('train_ratio')

            # Calculate error bars (95% confidence interval)
            yerr = 1.96 * model_data['pcc_std'] / np.sqrt(model_data['pcc_count'])

            # Plot the line with error bars
            plt.errorbar(
                model_data['train_ratio'] * 100,  # Convert to percentage
                model_data['pcc_mean'],
                yerr=yerr,
                label=f"{model}-{feature_type}",
                color=colors[i],
                marker=markers[j % len(markers)],
                markersize=8,
                capsize=5
            )

    # Add range information to the plot (similar to Nature paper)
    model_ranges = {}
    for model in models:
        # Calculate overall range across all feature types and train ratios
        model_data = grouped[grouped['model'] == model]
        if not model_data.empty:
            min_pcc = model_data['pcc_mean'].min()
            max_pcc = model_data['pcc_mean'].max()
            pcc_range = max_pcc - min_pcc
            model_ranges[model] = pcc_range

    # Add range text box
    range_text = "Range of each method:\n"
    for model, pcc_range in model_ranges.items():
        range_text += f"{model}: {pcc_range:.4f}  "

    plt.text(0.5, 0.05, range_text, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
             ha='center', fontsize=10)

    # Configure the plot
    plt.xlabel('Ratio of training set (%)')
    plt.ylabel('PCC', fontweight='bold')
    plt.title('Effect of Training Set Size on Model Performance', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=9)
    plt.xlim(0, 100)

    y_min = max(0, grouped['pcc_mean'].min() - 0.05)
    y_max = min(1.0, grouped['pcc_mean'].max() + 0.05)
    plt.ylim(y_min, y_max)

    # Add x ticks at each 10%
    plt.xticks(np.arange(0, 101, 10))

    # Save the plot
    plot_path = os.path.join(plots_dir, f"training_ratio_curve_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {plot_path}")

    # Return grouped results
    return grouped


def visualize_pdb_limited_results(results, output_dir):
    'Visualize pdb limited results.'
    if not results:
        print("No results are available for visualization.")
        return


    df = pd.DataFrame(results)


    df['rmse'] = np.sqrt(df['mse'])


    os.makedirs(output_dir, exist_ok=True)


    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)


    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f"pdb_limited_raw_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved to {csv_path}")


    summary = df.groupby(['model', 'feature_type', 'feature_name']).agg({
        'pcc': ['mean', 'std', 'count'],
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std']
    }).reset_index()


    summary.columns = ['model', 'feature_type', 'feature_name',
                       'pcc_mean', 'pcc_std', 'run_count',
                       'mse_mean', 'mse_std',
                       'mae_mean', 'mae_std',
                       'rmse_mean', 'rmse_std']


    summary_path = os.path.join(output_dir, f"pdb_limited_summary_{timestamp}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


    print("\nPDB-limited experiment summary (PCC):")
    for _, row in summary.iterrows():
        print(
            f"{row['model']} + {row['feature_name']}: {row['pcc_mean']:.4f} ± {row['pcc_std']:.4f} ({row['run_count']} runs)")


    comparison_path = os.path.join(plots_dir, f"pdb_limited_comparison_{timestamp}.png")
    create_pdb_limited_comparison_chart(
        data=summary,
        save_path=comparison_path,
        title=None,
        y_min=0.5,
        y_max=0.8,
        figsize=(16, 15),
    )
    print(f"PDB-limited comparison figure saved to {comparison_path}")


    boxplot_path = os.path.join(plots_dir, f"pdb_limited_boxplot_{timestamp}.png")
    create_publication_boxplot(
        data=df,
        x='model',
        y='pcc',
        hue='feature_name',
        save_path=boxplot_path,
        # title='Performance Distribution Across Models',
        showpoints=True,
        y_min=0.5,
        y_max=0.8,
        figsize=(16, 15),
    )
    print(f"Performance box plot saved to {boxplot_path}")

    return summary

def create_pdb_limited_comparison_chart(data, save_path, title=None, figsize=(16, 15), y_min=0.6, y_max=None):
    'Create pdb limited comparison chart.'

    plt.style.use('default')
    line_width = 1.6
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 28,
        'axes.linewidth': line_width,
        'axes.labelsize': 28,
        'axes.titlesize': 28,
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'font.weight': 'normal',
    })


    fig, ax = plt.subplots(figsize=figsize, dpi=150)


    model_name_map = {
        'simplified': 'iSCALE',
        'ssd': 'iSCALE*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN'
    }


    feature_name_map = {
        'Distribution only': 'Distribution',
        'No features': 'Baseline',
        'Intensity only': 'Intensity',
        'Full features': 'Both'
    }


    data_plot = data.copy()
    data_plot['feature_display'] = data_plot['feature_name'].map(lambda x: feature_name_map.get(x, x))


    best_per_model = data_plot.groupby('model')['pcc_mean'].max().reset_index()
    best_per_model = best_per_model.sort_values('pcc_mean', ascending=False)
    sorted_models = best_per_model['model'].tolist()


    model_display_names = [model_name_map.get(m, m) for m in sorted_models]


    data_plot['model'] = pd.Categorical(data_plot['model'], categories=sorted_models, ordered=True)
    data_plot = data_plot.sort_values('model')


    feature_names = ['Baseline', 'Distribution', 'Intensity', 'Both']

    feature_names = [f for f in feature_names if f in data_plot['feature_display'].unique()]
    models = data_plot['model'].unique()


    colors = ['#2E86AB', '#F18F01']


    n_features = len(feature_names)
    bar_width = 0.70 / n_features
    spacing_factor = 0.9
    positions = np.arange(len(models)) * spacing_factor


    max_val = data_plot['pcc_mean'].max()
    if y_max is None:
        y_max = min(1.0, max_val * 1.05)
    ax.set_ylim(y_min, y_max)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    ax.spines['bottom'].set_linewidth(line_width * 1.1)
    ax.spines['left'].set_linewidth(line_width * 1.1)


    ax.spines['bottom'].set_zorder(5)
    ax.spines['left'].set_zorder(5)


    ax.grid(False)


    bars_dict = {}
    for i, feature in enumerate(feature_names):
        feature_data = data_plot[data_plot['feature_display'] == feature]
        feature_data = feature_data.set_index('model').reindex(models)


        offset = (i - n_features / 2 + 0.5) * bar_width
        x_pos = positions + offset


        bars = ax.bar(x_pos, feature_data['pcc_mean'],
                      width=bar_width,
                      color=colors[i],
                      edgecolor='black',
                      linewidth=line_width * 0.9,
                      alpha=0.92,
                      label=feature,
                      zorder=3)
        bars_dict[feature] = bars


        if 'pcc_std' in feature_data.columns:
            ax.errorbar(x_pos, feature_data['pcc_mean'],
                        yerr=feature_data['pcc_std'],
                        fmt='none',
                        ecolor='black',
                        elinewidth=1.2,
                        capsize=4,
                        zorder=4)


        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):

                bar_color = bar.get_facecolor()



                r, g, b, a = bar_color
                brightness = 0.299 * r + 0.587 * g + 0.114 * b


                text_color = 'white' if brightness < 0.55 else 'black'


                bar_bottom = y_min


                text_offset = 0.008
                y_pos = bar_bottom + text_offset


                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=24,
                        fontweight='normal',
                        color=text_color,
                        rotation=90,

                        path_effects=[path_effects.Stroke(linewidth=0.2, foreground=text_color),
                                      path_effects.Normal()],
                        zorder=5)


    ax.set_xticks(positions)


    ax.set_xlim(-0.5, len(models) * spacing_factor - 0.5)
    ax.set_xticklabels(model_display_names, rotation=40, ha='right')


    ax.tick_params(axis='both', which='both', zorder=6)


    ax.set_xlabel('')


    ax.set_ylabel('PCC', fontsize=28, fontweight='normal', labelpad=18)


    if title:
        ax.set_title('Model Performance with Different Features', fontsize=30, fontweight='normal', pad=12)


    legend = ax.legend(
        frameon=True,
        framealpha=0.75,
        edgecolor='#555555',
        fontsize=26,
        loc='upper right'
    )


    plt.tight_layout(pad=1.5)


    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return fig, ax

def create_publication_boxplot(data, x, y, hue, save_path, title=None, figsize=(16, 15),
                               palette=None, showfliers=True, showpoints=True, y_min=None, y_max=None):
    """
    Create a publication-quality box plot with top-tier journal styling

    Parameters:
        data: DataFrame with the data
        x: Column name for x-axis categories
        y: Column name for y-axis values
        hue: Column name for grouping/coloring
        save_path: Path to save the figure
        title: Title for the plot
        figsize: Figure size
        palette: Color palette
        showfliers: Whether to show outliers
        showpoints: Whether to show individual data points
        y_min, y_max: Y-axis limits (optional)
    """
    model_name_map = {
        'simplified': 'iSCALE',
        'ssd': 'iSCALE*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN'
    }

    feature_name_map = {
        'Distribution only': 'Distribution',
        'No features': 'Baseline',
        'Intensity only': 'Intensity',
        'Full features': 'Both'
    }


    data_mapped = data.copy()


    if x in data_mapped.columns:
        data_mapped[x] = data_mapped[x].map(lambda x_val: model_name_map.get(x_val, x_val))


    if hue in data_mapped.columns:
        data_mapped[hue] = data_mapped[hue].map(lambda hue_val: feature_name_map.get(hue_val, hue_val))



        base_medians = []
        for model in data_mapped[x].unique():
            model_base_data = data_mapped[
                (data_mapped[x] == model) &
                (data_mapped[hue] == 'Base')
                ]

            if len(model_base_data) > 0:
                median_val = model_base_data[y].median()
            else:
                model_data = data_mapped[data_mapped[x] == model]
                median_val = model_data[y].median()

            base_medians.append((model, median_val))


        base_medians.sort(key=lambda x: x[1], reverse=True)
        sorted_models = [model for model, _ in base_medians]

        print("Models sorted by the median base-feature result:")
        for model, median in base_medians:
            print(f"  {model}: {median:.3f}")


        data_mapped[x] = pd.Categorical(data_mapped[x], categories=sorted_models, ordered=True)



        fixed_feature_order = ['Baseline', 'Distribution', 'Intensity', 'Both']
        existing_features = [f for f in fixed_feature_order if f in data_mapped[hue].unique()]
        data_mapped[hue] = pd.Categorical(data_mapped[hue], categories=existing_features, ordered=True)
        data_mapped = data_mapped.sort_values([x, hue])



    plt.style.use('default')
    line_width = 1.6
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 28,
        'axes.linewidth': line_width,
        'axes.labelsize': 28,
        'axes.titlesize': 28,
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'font.weight': 'normal',
    })

    # Set default palette if none provided
    if palette is None:
        palette = ['#2E86AB', '#F18F01']
        palette = sns.color_palette(palette,  len(data[hue].unique()))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Calculate data range for y-axis limits
    if y_min is None or y_max is None:
        y_data_min = data_mapped[y].min()
        y_data_max = data_mapped[y].max()

        if y_min is None:
            y_min = max(0, y_data_min - 0.05)
        if y_max is None:
            y_max = min(1.0, y_data_max + 0.05)

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Remove default grid
    ax.grid(False)

    # Create elegant box plot
    boxplot = sns.boxplot(
        data=data_mapped,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        width=0.8,
        fliersize=5 if showfliers else 0,
        linewidth=line_width * 0.9,
        whiskerprops=dict(linewidth=line_width * 1.2, color='black'),
        medianprops=dict(color='black', linewidth=line_width * 1.4),
        capprops=dict(linewidth=line_width * 1.2, color='black'),
        flierprops=dict(marker='+', markeredgecolor='black', markersize=5),
        ax=ax,
        zorder=3
    )

    # Add individual data points if requested
    if showpoints:
        # More elegant strip plot with jitter
        sns.stripplot(
            data=data_mapped,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            size=5,
            alpha=0.6,
            jitter=True,
            dodge=True,
            ax=ax,
            legend=False,
            zorder=2
        )

    for i, (group_name, group_data) in enumerate(data_mapped.groupby([x, hue])):
        x_val, hue_val = group_name

        # Find the category index in the plot
        x_index = list(data_mapped[x].unique()).index(x_val)
        hue_index = list(data_mapped[hue].unique()).index(hue_val)

        # Calculate the offset based on the hue
        n_hues = len(data_mapped[hue].unique())
        offset = (hue_index - (n_hues - 1) / 2) * (0.8 / n_hues)

        # Get the median and quartiles
        median = group_data[y].median()


        text_y_pos = median - 0.02 * (y_max - y_min)



        color = palette[hue_index]
        if isinstance(color, str):

            import matplotlib.colors as mcolors
            color = mcolors.to_rgba(color)


        r, g, b = color[:3]
        brightness = 0.299 * r + 0.587 * g + 0.114 * b


        text_color = 'white' if brightness < 0.55 else 'black'


        ax.text(
            x_index + offset,
            text_y_pos,
            f"{median:.2f}",
            ha='center',
            va='center',
            fontsize=17,
            fontweight='normal',
            color=text_color,
            path_effects=[path_effects.Stroke(linewidth=0.2, foreground=text_color),
                          path_effects.Normal()],
            zorder=6,
        )

    # Refine plot styling - enhance bottom border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(line_width * 1.1)
    ax.spines['left'].set_linewidth(line_width * 1.1)

    # Set layer order
    ax.spines['bottom'].set_zorder(5)
    ax.spines['left'].set_zorder(5)
    ax.tick_params(axis='both', which='both', zorder=6)

    # Enhance axis labels with normal weight
    ax.set_xlabel('', fontsize=28, fontweight='normal', labelpad=18)
    ax.set_ylabel('PCC', fontsize=28, fontweight='normal', labelpad=18)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')

    # Add title if provided - with refined styling
    if title:
        box_style = dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.9, edgecolor='#e0e0e0')
        ax.set_title(title, fontsize=30, fontweight='normal', pad=20, bbox=box_style)

    # Create more elegant legend
    legend = ax.legend(
        frameon=True,
        framealpha=0.8,
        edgecolor='#555555',
        fontsize=18,
        title_fontsize=24,
        loc='upper right'
    )


    legend.get_title().set_fontweight('normal')

    # Better spacing
    plt.tight_layout(pad=2.5)

    # Save with high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return fig, ax
# def create_publication_barplot(data, x, y, hue, save_path, title=None, figsize=(11, 8),
#                                yerr_col=None, palette=None):
#     """
#     Create a publication-quality bar plot matching your scatter plot style
#
#     Parameters:
#         data: DataFrame with the data
#         x: Column name for x-axis categories
#         y: Column name for y-axis values
#         hue: Column name for grouping/coloring
#         save_path: Path to save the figure
#         title: Title for the plot
#         figsize: Figure size
#         yerr_col: Column name for error bars
#         palette: Color palette (if None, uses a blue palette)
#     """
#     # Set style to match your scatter plot
#     plt.style.use('default')
#     plt.rcParams.update({
#         'font.family': ['serif'],
#         'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
#         'font.size': 12,
#         'axes.linewidth': 2,
#         'axes.labelsize': 16,
#         'axes.titlesize': 16,
#         'xtick.major.width': 2,
#         'ytick.major.width': 2,
#         'xtick.major.size': 5,
#         'ytick.major.size': 5,
#         'xtick.labelsize': 14,
#         'ytick.labelsize': 14,
#         'font.weight': 'bold',
#     })
#
#     # Set default palette if none provided
#     if palette is None:
#         palette = sns.color_palette("Blues_d", len(data[hue].unique()))
#
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize, dpi=150)
#
#     # Create bar plot with Seaborn for consistent styling
#     bars = sns.barplot(
#         data=data,
#         x=x,
#         y=y,
#         hue=hue,
#         palette=palette,
#         errorbar=None,  # We'll add custom error bars
#         ax=ax
#     )
#
#     # Add custom error bars if specified
#     if yerr_col is not None:
#         # Get the number of groups and categories
#         n_groups = len(data[hue].unique())
#         n_categories = len(data[x].unique())
#
#         # Calculate positions for bars
#         for i, (_group_index, group_data) in enumerate(data.groupby([hue])):
#             for j, (_cat_index, cat_data) in enumerate(group_data.groupby([x])):
#                 # Calculate bar position
#                 index = j
#                 # Adjust position based on group
#                 group_offset = (i - (n_groups - 1) / 2) * (0.8 / n_groups)
#
#                 # Draw error bar
#                 if not cat_data[yerr_col].isna().all():
#                     yerr = cat_data[yerr_col].values[0]
#                     ax.errorbar(
#                         x=index + group_offset,
#                         y=cat_data[y].values[0],
#                         yerr=yerr,
#                         fmt='none',
#                         ecolor='black',
#                         elinewidth=1.5,
#                         capsize=4,
#                         capthick=1.5,
#                         zorder=10
#                     )
#
#                     # Add value labels on top of bars
#                     ax.text(
#                         index + group_offset,
#                         cat_data[y].values[0] + yerr + 0.01,
#                         f"{cat_data[y].values[0]:.4f}",
#                         ha='center',
#                         va='bottom',
#                         fontsize=9,
#                         fontweight='bold',
#                         color='black',
#                         rotation=45
#                     )
#
#     # Refine plot styling
#
#     # Remove top and right spines (like your scatter plot)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
#     # Add subtle grid for y-axis only
#     ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
#
#     # Enhance axis labels
#     ax.set_xlabel(x, fontsize=16, fontweight='bold', labelpad=10)
#     ax.set_ylabel(y, fontsize=16, fontweight='bold', labelpad=10)
#
#     # Add title if provided
#     if title:
#         ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
#
#     # Create more elegant legend
#     legend = ax.legend(
#         title=hue,
#         frameon=True,
#         framealpha=0.9,
#         edgecolor='black',
#         fontsize=12,
#         title_fontsize=14,
#         loc='upper right'
#     )
#
#     # Better spacing
#     plt.tight_layout(pad=2.0)
#
#     # Save with high quality
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
#     plt.close(fig)
#
#     return fig, ax

def create_publication_heatmap(data, save_path, title=None, figsize=(12, 8),
                               cmap='RdBu_r', annot_fmt='.4f', vmin=None, vmax=None,
                               custom_order=None, aspect=25):
    'Create publication heatmap.'


    name_mapping = {
        'simplified': 'iSCALE',
        'ssd': 'iSCALE*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN',
    }


    feature_mapping = {
        'No features': 'Baseline',
        'Distribution only': 'Distribution',
        'Intensity only': 'Intensity',
        'Full features': 'All',
    }


    threshold_mapping = {
        'Baseline': 'Baseline',
        'fine': 'Fine-grained',
        'near': 'Near-range',
        'dense': 'Dense',
        'hydrophobic': 'Hydrophobic',
        'default': 'Contact',
        'electrostatic': 'Electrostatic',
        'domain': 'Domain',
        'sparse': 'Sparse',
        'coarse': 'Coarse-grained'
    }


    print("Input row labels:", data.index.tolist())
    print("Input columns:", data.columns.tolist())


    data_mapped = data.copy()
    data_mapped.index = [name_mapping.get(idx, idx) for idx in data.index]

    print("Mapped row labels:", data_mapped.index.tolist())
    print("Mapped columns:", data_mapped.columns.tolist())


    has_feature_data = any(col in feature_mapping for col in data_mapped.columns)
    has_threshold_data = any(col in threshold_mapping for col in data_mapped.columns)

    if has_feature_data:

        data_mapped.columns = [feature_mapping.get(col, col) for col in data_mapped.columns]


        available_columns = data_mapped.columns.tolist()
        feature_order = list(feature_mapping.values())


        ordered_columns = [col for col in feature_order if col in available_columns]

        missing_columns = [col for col in available_columns if col not in ordered_columns]
        final_column_order = ordered_columns + missing_columns

        print("Feature data detected; applying the feature label order.")
        print("Final column order:", final_column_order)
        data_mapped = data_mapped.reindex(columns=final_column_order)

    elif has_threshold_data:

        data_mapped.columns = [threshold_mapping.get(col, col) for col in data_mapped.columns]


        available_columns = data_mapped.columns.tolist()
        threshold_order = list(threshold_mapping.values())


        ordered_columns = [col for col in threshold_order if col in available_columns]

        missing_columns = [col for col in available_columns if col not in ordered_columns]
        final_column_order = ordered_columns + missing_columns

        print("Threshold data detected; applying the threshold order.")
        print("Final column order:", final_column_order)
        data_mapped = data_mapped.reindex(columns=final_column_order)

    else:

        print("No feature or threshold labels detected; preserving input order.")
        print("Final column order:", data_mapped.columns.tolist())


    if custom_order is not None:
        print("Requested custom row order:", custom_order)


        available_models = data_mapped.index.tolist()
        print("Available models:", available_models)


        valid_order = [model for model in custom_order if model in available_models]
        print("Valid custom order:", valid_order)


        missing_models = [model for model in available_models if model not in valid_order]
        final_order = valid_order + missing_models
        print("Final row order:", final_order)


        data_mapped = data_mapped.reindex(final_order)


    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif'],
        'font.size': 24,
        'axes.linewidth': 1.2,
    })


    fig, ax = plt.subplots(figsize=figsize, dpi=300)


    hm = sns.heatmap(
        data_mapped,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        linewidths=1,
        linecolor='white',
        cbar_kws={
            'shrink': 1.0,
            'aspect': aspect,
            'pad': 0.02,
            'ticks': [0.8, 0.85, 0.9, 0.95],
        },
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )


    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)

    for i, j in np.ndindex(data_mapped.shape):
        if j < len(data_mapped.columns) and i < len(data_mapped.index):

            try:
                value = data_mapped.iloc[i, j]
                if not np.isnan(value):

                    rgba = cmap_obj(norm(value))

                    brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


                    idx = i * len(data_mapped.columns) + j
                    if idx < len(ax.texts):
                        text = ax.texts[idx]

                        if brightness < 0.50:
                            text.set_color('white')
                        else:
                            text.set_color('black')


                        text.set_fontsize(20)
            except (IndexError, ValueError):
                pass


    # if title is None:
    #     title = "PCC performance of different models in 5-fold cross-validation"
    ax.set_title(title, fontsize=24, fontweight='normal', pad=15)


    plt.setp(ax.get_xticklabels(),
             rotation=40, ha='right',
             fontsize=20, fontweight='normal')
    plt.setp(ax.get_yticklabels(),
             rotation=0,
             fontsize=20, fontweight='normal')


    cbar = ax.collections[0].colorbar



    cbar.ax.tick_params(
        labelsize=20,
        colors='black',
        width=1.2,
        length=4
    )


    plt.tight_layout()


    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"Heat map saved to {save_path}")
    return fig, ax

def visualize_results(results, output_dir):
    'Visualize results.'
    if not results:
        print("No results are available for visualization.")
        return


    df = pd.DataFrame(results)


    pcc_pivot = pd.pivot_table(df, values='pcc', index=['model'], columns=['feature_name'])


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')


    df.to_csv(os.path.join(output_dir, f"experiment_matrix_results_{timestamp}.csv"), index=False)


    heatmap_path = os.path.join(output_dir, f"publication_heatmap_{timestamp}.png")
    create_publication_heatmap(
        data=pcc_pivot,
        save_path=heatmap_path,
        title="PCC values of all methods on RNA-binding prediction",

        vmin=0.75,
        vmax=0.95
    )


    print("\nExperiment-matrix summary (PCC):")
    print(pcc_pivot.round(4))

    return pcc_pivot

def plot_training_ratio_curves(results_df, output_path, title=None, figsize=(10, 6), y_min=None, y_max=None):
    """
    Plot a simple line graph showing PCC vs training ratio for different methods.

    Parameters:
        results_df: DataFrame with columns 'model', 'train_ratio', and 'pcc'
        output_path: Path to save the output figure
        title: Optional title for the plot
        figsize: Figure size as (width, height) tuple
        y_min, y_max: Optional y-axis limits

    Returns:
        fig, ax: The created figure and axis objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique models
    models = results_df['model'].unique()

    # Define a colormap and markers
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h']

    # Calculate model ranges for the table
    model_ranges = {}

    # Plot each model as a separate line
    for i, model in enumerate(models):
        # Filter data for this model
        model_data = results_df[results_df['model'] == model]

        # Sort by train_ratio
        model_data = model_data.sort_values('train_ratio')

        # Get color and marker
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # Plot the line
        ax.plot(
            model_data['train_ratio'] * 100,  # Convert to percentage
            model_data['pcc'],
            label=model,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2
        )

        # Calculate range
        if len(model_data) > 1:
            min_pcc = model_data['pcc'].min()
            max_pcc = model_data['pcc'].max()
            pcc_range = max_pcc - min_pcc
            model_ranges[model] = pcc_range

    # Add range information as a text box
    if model_ranges:
        range_text = "Range of each method:\n"
        for model, pcc_range in sorted(model_ranges.items(), key=lambda x: -x[1]):
            range_text += f"{model}: {pcc_range:.4f}  "

        ax.text(0.5, 0.01, range_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                ha='center', fontsize=10)

    # Configure the plot
    ax.set_xlabel('Ratio of training set (%)')
    ax.set_ylabel('PCC')

    if title:
        ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='best')

    # Set x-axis limits and ticks
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))

    # Set y-axis limits if provided
    if y_min is not None or y_max is not None:
        y_min = y_min if y_min is not None else ax.get_ylim()[0]
        y_max = y_max if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(y_min, y_max)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Also save as PDF for publication quality
    pdf_path = output_path.replace('.png', '.pdf') if output_path.endswith('.png') else output_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')

    return fig, ax


def plot_publication_training_ratio_v0(results_df, output_path, figsize=(10, 6), y_min=None, y_max=1.0):
    """
    Create a refined, elegant publication-quality plot showing PCC vs training ratio.
    Follows high-end scientific journal aesthetics with meticulous design details.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl

    # Set style for refined scientific plots
    plt.style.use('default')
    line_width = 1.5  # Unified line width

    # Define axis color - slightly transparent dark gray for the axis lines only
    axis_color = (0.2, 0.2, 0.2, 0.85)  # RGBA with alpha

    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.linewidth': line_width,
        'axes.labelsize': 13,
        'axes.titlesize': 16,
        'axes.edgecolor': axis_color,
        'xtick.major.width': line_width,
        'xtick.major.width': line_width,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Create figure and axis with white background
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')

    # Get unique models
    models = results_df['model'].unique()

    # INTERNAL NAME MAPPING - MODIFY HERE
    # Map internal model names to display names
    name_mapping = {
        'simplified': 'iSCALE',
        'ssd': 'iSCALE*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN',
        'mamba_triple': 'Mamba',
        # Add/modify as needed
    }

    # Define fixed marker and color mappings for common models with more distinct colors
    marker_map = {
        'simplified': 'o',  # Circle
        'ssd': 'o',  # Circle
        'transformer': 's',  # Square
        'graph_transformer': 'h',  # Hexagon - different from transformer
        'gcn': '^',  # Triangle up
        'gat': 'D',  # Diamond
        'gin': 'P',  # Plus (filled)
        'sage': '*',  # Star
        'edge': 'X',  # X (filled)
        'mamba_triple': 'p',  # Pentagon
    }

    # Revised color map with more distinct colors
    color_map = {
        'simplified': '#4878D0',  # Blue
        'ssd': '#82C6E2',  # Light blue
        'transformer': '#EE854A',  # Orange
        'graph': '#6A5ACD',  # Slate blue
        'gcn': '#6ACC64',  # Green
        'gat': '#D65F5F',  # Red
        'gin': '#956CB4',  # Purple
        'sage': '#8C613C',  # Brown
        'edge': '#DC7EC0',  # Magenta
        'mamba_triple': '#FFD700',  # Gold
    }

    # Default markers and colors for any unlisted models
    default_markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'h', 'p']
    default_colors = ['#4878D0', '#EE854A', '#6ACC64', '#D65F5F', '#956CB4', '#8C613C', '#DC7EC0', '#82C6E2', '#6A5ACD']

    # Calculate the max performance for each model to sort by performance
    model_performance = {}
    model_ranges = {}

    for model in models:
        model_data = results_df[results_df['model'] == model]
        if not model_data.empty:
            max_pcc = model_data['pcc'].max()
            model_performance[model] = max_pcc

            # Also calculate range if there are multiple points
            if len(model_data) > 1:
                min_pcc = model_data['pcc'].min()
                pcc_range = max_pcc - min_pcc
                model_ranges[model] = pcc_range

    # Sort models by max performance (descending)
    sorted_models = sorted(models, key=lambda m: model_performance.get(m, 0), reverse=True)

    # Line objects for legend
    lines = []
    labels = []

    # Plot each model as a separate line
    for i, model in enumerate(sorted_models):
        # Filter data for this model
        model_data = results_df[results_df['model'] == model].sort_values('train_ratio')

        # Get marker and color (from map or default)
        marker = marker_map.get(model, default_markers[i % len(default_markers)])
        color = color_map.get(model, default_colors[i % len(default_colors)])

        # Get display name
        display_name = name_mapping.get(model, model)

        # Plot the line with semi-transparency
        line, = ax.plot(
            model_data['train_ratio'] * 100,  # Convert to percentage
            model_data['pcc'],
            label=display_name,  # Use display name in legend
            color=color,
            marker=marker,
            markersize=7,
            markeredgecolor='white',
            markeredgewidth=1,
            linewidth=2,
            alpha=0.85,
            zorder=3 + (len(models) - i)  # Higher performance models on top
        )

        lines.append(line)
        labels.append(display_name)  # Use display name for labels

    # Configure the plot with refined axis labels
    ax.set_xlabel('Ratio of training set (%)', fontweight='normal')  # Lighter weight
    ax.set_ylabel('PCC', fontweight='normal')  # Lighter weight

    # Set x-axis limits and ticks - extend to 100 but don't label 100
    ax.set_xlim(0, 100)
    xticks = np.arange(0, 100, 10)  # 0 to 90 by 10
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], rotation=30, ha='right')  # Rotate 30 degrees

    # Set y-axis limits
    if y_min is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(ax.get_ylim()[0], y_max)

    # Remove gridlines
    ax.grid(False)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make tick labels (numbers) not bold
    for tick in ax.get_xticklabels():
        tick.set_fontweight('normal')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('normal')

    # Create a legend in upper left with increased spacing and larger font
    legend = ax.legend(
        lines, labels,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),  # Position in upper left
        ncol=3,  # 3 columns for 9 methods
        frameon=False,  # No frame/border around legend
        fontsize=12,  # Larger font
        handlelength=1.2,
        columnspacing=1.8,  # Increased column spacing
        labelspacing=0.9,  # Increased row spacing
    )

    # Add range information title and table
    if model_ranges:
        # Define the range box coordinates
        range_box_left = 0.26
        range_box_right = 0.94
        range_box_bottom = 0.02
        range_box_top = 0.20

        # Position for the title (outside the box, centered above the range box)
        ax.text((range_box_left + range_box_right) / 2, range_box_top + 0.02,
                "Range of each method",
                transform=ax.transAxes,
                ha='center', va='bottom',
                fontsize=13, fontweight='bold')

        # Precisely create the range table
        # Creating a compact table directly in the axis coordinates
        # Get all displayed models in their sorted order
        range_data = []
        for model in sorted_models:
            if model in model_ranges:
                display_name = name_mapping.get(model, model)
                range_data.append((display_name, model_ranges[model]))

        # Make sure we have 9 items (3x3 grid)
        while len(range_data) < 9:
            range_data.append((None, None))

        # Create a 3x3 grid layout
        rows = []
        for i in range(0, 9, 3):
            row = range_data[i:i + 3]
            rows.append(row)

        # Draw a dotted rectangle for the table
        import matplotlib.patches as patches
        rect = patches.Rectangle(
            (range_box_left, range_box_bottom),  # (x, y) of bottom left corner
            range_box_right - range_box_left,  # width
            range_box_top - range_box_bottom,  # height
            transform=ax.transAxes,
            fill=True,
            facecolor='#f9f9f9',  # VERY light gray
            edgecolor='black',
            linestyle='dotted',
            linewidth=1,
            alpha=1.0,  # Fully opaque
            zorder=1
        )
        ax.add_patch(rect)

        # Add horizontal separators using plt.plot
        for y_pos in [0.07, 0.13]:  # Adjusted for the box height
            ax.plot(
                [range_box_left, range_box_right],  # x-coordinates
                [y_pos, y_pos],  # y-coordinates: flat line
                color='gray',
                linestyle='-',
                linewidth=0.5,
                alpha=0.5,
                transform=ax.transAxes
            )

        # Calculate column positions
        col_1 = range_box_left + 0.02
        col_2 = (range_box_left + range_box_right) / 2 - 0.10
        col_3 = range_box_right - 0.22

        # Row positions
        row_1 = 0.16
        row_2 = 0.10
        row_3 = 0.04
        textfont = 11

        # Add the text entries with manual positioning for perfect alignment
        # First row
        ax.text(col_1, row_1, f"{rows[0][0][0]}: {rows[0][0][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_2, row_1, f"{rows[0][1][0]}: {rows[0][1][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_3, row_1, f"{rows[0][2][0]}: {rows[0][2][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')

        # Second row
        ax.text(col_1, row_2, f"{rows[1][0][0]}: {rows[1][0][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_2, row_2, f"{rows[1][1][0]}: {rows[1][1][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_3, row_2, f"{rows[1][2][0]}: {rows[1][2][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')

        # Third row
        ax.text(col_1, row_3, f"{rows[2][0][0]}: {rows[2][0][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_2, row_3, f"{rows[2][1][0]}: {rows[2][1][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')
        ax.text(col_3, row_3, f"{rows[2][2][0]}: {rows[2][2][1]:.4f}", transform=ax.transAxes, fontsize=textfont, ha='left',
                va='center')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Also save as PDF for publication quality
    pdf_path = output_path.replace('.png', '.pdf') if output_path.endswith('.png') else output_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')

    return fig, ax


def plot_publication_training_ratio(results_df, output_path, figsize=(12, 8), y_min=None, y_max=1.0,
                                    manual_offsets=None):
    """
    Plot PCC as a function of the training-set ratio using uniform styling.

    Parameters:
        results_df: DataFrame with columns ['model', 'train_ratio', 'pcc']
        output_path: Path to save the figure
        figsize: Figure size tuple
        y_min, y_max: Y-axis limits
        manual_offsets: Retained for compatibility; no method-specific annotation is applied.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl


    plt.style.use('default')
    line_width = 1.8

    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 24,
        'axes.linewidth': 1.2,
        'axes.labelsize': 28,
        'axes.titlesize': 30,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.labelsize': 26,
        'ytick.labelsize': 26,
    })

    # Use the same visual weight for every method.
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi=150)
    ax.set_facecolor('white')

    models = results_df['model'].unique()

    name_mapping = {
        'simplified': 'iSCALE',
        'ssd': 'iSCALE*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN',
        'mamba_triple': 'Mamba',
    }

    marker_map = {
        'simplified': 'o',
        'ssd': 's',
        'transformer': '^',
        'graph': 'D',
        'gcn': 'v',
        'gat': 'p',
        'gin': '*',
        'sage': 'h',
        'edge': 'X',
        'mamba_triple': 'P',
    }

    color_map = {
        'simplified': '#E63946',
        'ssd': '#457B9D',
        'transformer': '#F77F00',
        'graph': '#6A994E',
        'gcn': '#A663CC',
        'gat': '#F72585',
        'gin': '#4CC9F0',
        'sage': '#7209B7',
        'edge': '#FB8500',
        'mamba_triple': '#219EBC',
    }

    sorted_models = sorted(models, key=lambda model: name_mapping.get(model, model).lower())

    # Plot each model
    lines = []
    labels = []
    model_line_map = {}

    for i, model in enumerate(sorted_models):
        model_data = results_df[results_df['model'] == model].sort_values('train_ratio')

        marker = marker_map.get(model, 'o')
        color = color_map.get(model, '#666666')
        display_name = name_mapping.get(model, model)

        line, = ax.plot(
            model_data['train_ratio'] * 100,
            model_data['pcc'],
            label=display_name,
            color=color,
            marker=marker,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.2,
            linewidth=2.5,
            alpha=0.9,
            zorder=5,
        )

        lines.append(line)
        labels.append(display_name)
        model_line_map[model] = line

    # Enhanced axis configuration
    ax.set_xlabel('Training Set Ratio (%)', fontweight='normal', fontsize=28)
    ax.set_ylabel('PCC', fontweight='normal', fontsize=28)

    # Enhanced x-axis
    ax.set_xlim(0, 100)
    xticks = np.arange(0, 100, 10)
    ax.set_xticks(xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks])

    # Set y-axis limits
    if y_min is not None:
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(ax.get_ylim()[0], y_max)

    # Remove gridlines and spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)

    legend_elements = []
    legend_labels = []

    for model in sorted_models:
        line = model_line_map[model]
        display_name = name_mapping.get(model, model)
        legend_elements.append(line)
        legend_labels.append(display_name)

    ax.legend(
        legend_elements, legend_labels,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fontsize=18,
        handlelength=2.0,
        columnspacing=1.5,
        labelspacing=0.8,
        title='Methods',
        title_fontsize=18,
    )

    # Enhanced layout
    plt.tight_layout(pad=1.0)

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = output_path.replace('.png', '.pdf') if output_path.endswith('.png') else output_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    return fig, ax
