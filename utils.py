# utils.py - 多尺度特征实验辅助工具集
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


# ============= 特征维度计算功能 =============

# 1. 🔥 更新现有的 calculate_feature_dimensions 函数
def calculate_feature_dimensions(base_protein_channels, base_rna_channels, feature_type, contact_thresholds=None):
    """
    根据特征类型计算正确的特征维度 - 更新版本支持ESM2特征

    参数:
        base_protein_channels: 基础蛋白质特征维度 (通常为41)
        base_rna_channels: 基础RNA特征维度 (通常为5)
        feature_type: 特征类型 (0-7)
            0=无多尺度特征, 1=仅分布特征, 2=仅强度特征, 3=完整多尺度特征
            4=ESM2特征, 5=ESM2+分布特征, 6=ESM2+强度特征, 7=ESM2+完整多尺度特征
        contact_thresholds: 接触阈值列表，默认使用配置中的默认值

    返回:
        protein_channels, rna_channels: 计算后的特征维度
    """
    if contact_thresholds is None:
        contact_thresholds = DEFAULT_CONTACT_THRESHOLDS

    # ESM2特征维度
    esm2_dim = 1280  # ESM2-650M的特征维度

    # 计算接触特征维度
    contact_dist_dim = len(contact_thresholds) + 1  # 分布特征维度
    contact_int_dim = 1  # 强度特征维度

    # 根据特征类型计算维度
    if feature_type == 0:
        # 无多尺度特征 - 原有逻辑保持不变
        protein_channels = base_protein_channels
        rna_channels = base_rna_channels

    elif feature_type == 1:
        # 仅分布特征 - 原有逻辑保持不变
        protein_channels = base_protein_channels + contact_dist_dim
        rna_channels = base_rna_channels + contact_dist_dim

    elif feature_type == 2:
        # 仅强度特征 - 原有逻辑保持不变
        protein_channels = base_protein_channels + contact_int_dim
        rna_channels = base_rna_channels + contact_int_dim

    elif feature_type == 3:
        # 完整多尺度特征 - 原有逻辑保持不变
        protein_channels = base_protein_channels + contact_dist_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_dist_dim + contact_int_dim

    elif feature_type == 4:
        # ESM2特征（纯大模型特征）
        protein_channels = esm2_dim
        rna_channels = base_rna_channels  # RNA仍使用基础特征

    elif feature_type == 5:
        # ESM2+分布特征
        protein_channels = esm2_dim + contact_dist_dim
        rna_channels = base_rna_channels + contact_dist_dim

    elif feature_type == 6:
        # ESM2+强度特征
        protein_channels = esm2_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_int_dim

    elif feature_type == 7:
        # ESM2+完整多尺度特征
        protein_channels = esm2_dim + contact_dist_dim + contact_int_dim
        rna_channels = base_rna_channels + contact_dist_dim + contact_int_dim

    else:
        # 未知特征类型，使用原有逻辑作为后备
        print(f"警告: 未知的特征类型 {feature_type}，使用基础特征")
        protein_channels = base_protein_channels
        rna_channels = base_rna_channels

    return protein_channels, rna_channels


# 2. 🔥 更新现有的 get_feature_type_name 函数
def get_feature_type_name(feature_type):
    """获取特征类型的描述性名称 - 更新版本支持ESM2特征"""
    feature_type_names = {
        # 原有特征类型保持不变
        0: "无多尺度特征",
        1: "仅使用分布特征",
        2: "仅使用强度特征",
        3: "完整多尺度特征",

        # 新增ESM2特征类型
        4: "ESM2特征",
        5: "ESM2+分布特征",
        6: "ESM2+强度特征",
        7: "ESM2+完整多尺度特征"
    }
    return feature_type_names.get(feature_type, f"未知特征类型({feature_type})")


# 3. 🔥 添加新的辅助函数
def is_esm2_feature_type(feature_type):
    """检查是否为ESM2特征类型"""
    return feature_type in [4, 5, 6, 7]


def get_feature_type_details(feature_type, contact_thresholds=None):
    """
    获取特征类型的详细信息

    Args:
        feature_type: 特征类型
        contact_thresholds: 接触阈值列表

    Returns:
        dict: 包含特征类型详细信息的字典
    """
    if contact_thresholds is None:
        contact_thresholds = DEFAULT_CONTACT_THRESHOLDS

    # 基础维度
    base_protein_dim = 41
    base_rna_dim = 5
    esm2_dim = 1280
    contact_dist_dim = len(contact_thresholds) + 1
    contact_int_dim = 1

    # 计算维度
    protein_dim, rna_dim = calculate_feature_dimensions(
        base_protein_dim, base_rna_dim, feature_type, contact_thresholds
    )

    # 组成成分
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
    """获取特定模型的参数覆盖"""
    model_params = {
        # 主要模型
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

    # 返回特定模型参数，如果没有则返回空字典
    return model_params.get(model_name, {})


# ============= 交叉验证功能 =============
def random_kfold_split(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    """
    随机k折交叉验证数据划分，不考虑PDB分组

    参数:
        data_list: 数据列表
        k: 折数，默认为5
        seed: 随机种子，确保可复现
        output_dir: 输出目录，用于保存划分结果
        visualize: 是否生成可视化图表

    返回:
        splits: 一个列表，包含k个(train_indices, val_indices)元组
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating random {k}-fold cross-validation split")
    print(f"Total samples: {len(data_list)}")

    # 生成所有索引并随机打乱
    indices = list(range(len(data_list)))
    random.shuffle(indices)

    # 计算每个fold的大小
    fold_size = len(indices) // k
    remainder = len(indices) % k

    # 分配索引到各个fold
    fold_indices = []
    start = 0
    for i in range(k):
        # 如果有余数，前remainder个fold多分配一个样本
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        fold_indices.append(indices[start:end])
        start = end

    # 创建训练/验证划分
    splits = []
    fold_sample_counts = [len(fold) for fold in fold_indices]

    for i in range(k):
        val_indices = fold_indices[i]
        train_indices = [idx for j, fold in enumerate(fold_indices) if j != i for idx in fold]
        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")

    # 统计各fold中的PDB分布情况（仅用于信息记录）
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

    # 保存划分信息
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存划分信息到文本文件
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

                # 添加PDB分布详情
                f.write(f"  PDB distribution in this fold:\n")
                for pdb, count in info['pdb_distribution'].items():
                    f.write(f"    {pdb}: {count} samples\n")
                f.write("\n")

        # 保存划分索引为NumPy格式，便于后续重用
        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )

    # 生成可视化
    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'Random {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')

        # 添加百分比标签
        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits


def pdb_based_kfold_split_with_randomness(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    """
    按PDB ID进行k折交叉验证数据划分，增加随机种子的影响
    确保同一PDB的样本不会同时出现在训练和验证集中
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating PDB-based {k}-fold cross-validation split with enhanced randomness")
    print(f"Total samples: {len(data_list)}")

    # 1. 按PDB ID分组
    pdb_groups = {}
    for idx, (wild_data, mutant_data, rna_data, ddg) in enumerate(data_list):
        pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(idx)

    # 获取所有PDB IDs和对应的样本数
    pdb_ids = list(pdb_groups.keys())

    # 简单修改1: 随机打乱PDB IDs的顺序
    random.shuffle(pdb_ids)

    pdb_sample_counts = {pdb: len(indices) for pdb, indices in pdb_groups.items()}

    print(f"Dataset contains {len(pdb_ids)} different PDB structures")

    # 2. 按样本数从大到小排序PDB，但引入随机性
    # 简单修改2: 不再严格按样本数排序，而是按样本数分组后随机排序
    size_groups = {}
    for pdb in pdb_ids:
        count = pdb_sample_counts[pdb]
        if count not in size_groups:
            size_groups[count] = []
        size_groups[count].append(pdb)

    # 将相同样本数的PDB随机排序
    pdb_by_size = []
    for count in sorted(size_groups.keys(), reverse=True):
        pdbs = size_groups[count]
        random.shuffle(pdbs)  # 随机排序相同样本数的PDB
        for pdb in pdbs:
            pdb_by_size.append((pdb, count))

    # 3. 初始化k个fold
    folds = [[] for _ in range(k)]
    fold_sample_counts = [0] * k

    # 4. 分配PDB到fold，使用贪心策略保持样本数平衡
    for pdb, count in pdb_by_size:
        # 找到当前样本数最少的fold
        min_fold_idx = fold_sample_counts.index(min(fold_sample_counts))
        folds[min_fold_idx].append(pdb)
        fold_sample_counts[min_fold_idx] += count

    # 5. 生成每个fold的详细信息
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

    # 6. 创建训练/验证划分
    splits = []
    for i in range(k):
        val_indices = []
        for pdb in folds[i]:
            val_indices.extend(pdb_groups[pdb])

        train_indices = []
        for j in range(k):
            if j != i:  # 除了当前fold外的所有fold作为训练集
                for pdb in folds[j]:
                    train_indices.extend(pdb_groups[pdb])

        # 简单修改3: 随机打乱训练集和验证集中的样本顺序
        random.shuffle(val_indices)
        random.shuffle(train_indices)

        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")

    # 7. 保存划分信息
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存划分信息到文本文件
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

        # 保存划分索引为NumPy格式，便于后续重用
        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )

    # 8. 生成可视化
    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'PDB-based {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')

        # 添加百分比标签
        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits

def pdb_based_kfold_split(data_list, k=5, seed=42, output_dir="./kfold_splits", visualize=True):
    """
    按PDB ID进行k折交叉验证数据划分，确保同一PDB的样本不会同时出现在训练和验证集中

    参数:
        data_list: 数据列表
        k: 折数，默认为5
        seed: 随机种子，确保可复现
        output_dir: 输出目录，用于保存划分结果
        visualize: 是否生成可视化图表

    返回:
        splits: 一个列表，包含k个(train_indices, val_indices)元组
    """
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Creating PDB-based {k}-fold cross-validation split")
    print(f"Total samples: {len(data_list)}")

    # 1. 按PDB ID分组
    pdb_groups = {}
    for idx, (wild_data, mutant_data, rna_data, ddg) in enumerate(data_list):
        pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
        if pdb_id not in pdb_groups:
            pdb_groups[pdb_id] = []
        pdb_groups[pdb_id].append(idx)

    # 获取所有PDB IDs和对应的样本数
    pdb_ids = list(pdb_groups.keys())
    pdb_sample_counts = {pdb: len(indices) for pdb, indices in pdb_groups.items()}

    print(f"Dataset contains {len(pdb_ids)} different PDB structures")

    # 2. 按样本数从大到小排序PDB，优先分配样本较多的PDB以保持平衡
    pdb_by_size = sorted(pdb_sample_counts.items(), key=lambda x: x[1], reverse=True)

    # 3. 初始化k个fold
    folds = [[] for _ in range(k)]
    fold_sample_counts = [0] * k

    # 4. 分配PDB到fold，使用贪心策略保持样本数平衡
    for pdb, count in pdb_by_size:
        # 找到当前样本数最少的fold
        min_fold_idx = fold_sample_counts.index(min(fold_sample_counts))
        folds[min_fold_idx].append(pdb)
        fold_sample_counts[min_fold_idx] += count

    # 5. 生成每个fold的详细信息
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

    # 6. 创建训练/验证划分
    splits = []
    for i in range(k):
        val_indices = []
        for pdb in folds[i]:
            val_indices.extend(pdb_groups[pdb])

        train_indices = []
        for j in range(k):
            if j != i:  # 除了当前fold外的所有fold作为训练集
                for pdb in folds[j]:
                    train_indices.extend(pdb_groups[pdb])

        splits.append((train_indices, val_indices))
        print(f"Split {i + 1}: Training {len(train_indices)} samples, Validation {len(val_indices)} samples")

    # 7. 保存划分信息
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 保存划分信息到文本文件
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

        # 保存划分索引为NumPy格式，便于后续重用
        np.savez(
            os.path.join(output_dir, 'fold_indices.npz'),
            **{f"train_fold_{i + 1}": train_indices for i, (train_indices, _) in enumerate(splits)},
            **{f"val_fold_{i + 1}": val_indices for i, (_, val_indices) in enumerate(splits)}
        )

    # 8. 生成可视化
    if visualize and output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, k + 1), fold_sample_counts, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Sample count')
        plt.title(f'PDB-based {k}-fold Cross Validation Sample Distribution')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, count in enumerate(fold_sample_counts):
            plt.text(i + 1, count + 5, str(count), ha='center')

        # 添加百分比标签
        for i, count in enumerate(fold_sample_counts):
            percentage = count / len(data_list) * 100
            plt.text(i + 1, count / 2, f"{percentage:.1f}%", ha='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fold_distribution.png'), dpi=300)
        plt.close()

    return splits


def load_fold_splits(npz_path):
    """加载已保存的fold划分"""
    data = np.load(npz_path)
    k = len([key for key in data.keys() if key.startswith('train_fold')])

    splits = []
    for i in range(k):
        train_indices = data[f'train_fold_{i + 1}']
        val_indices = data[f'val_fold_{i + 1}']
        splits.append((train_indices, val_indices))

    print(f"从 {npz_path} 加载了 {k} 个fold的划分")
    return splits


def create_fold_dataloaders(data_list, train_indices, val_indices, batch_size=16, num_workers=NUM_WORKERS,
                            prefetch_factor=PREFETCH_FACTOR):
    """
    根据索引创建训练和验证数据加载器

    参数:
        data_list: 原始数据列表
        train_indices: 训练集索引
        val_indices: 验证集索引
        batch_size: 批次大小
        num_workers: 工作进程数
        prefetch_factor: 预取因子

    返回:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 从原始数据集中提取训练和验证样本
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]

    # 创建数据加载器
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
    """
    可视化交叉验证结果

    参数:
        fold_results: 每个fold的结果列表
        avg_metrics: 平均指标
        output_dir: 输出目录
    """
    # 1. 创建性能对比柱状图
    plt.figure(figsize=(12, 6))

    # 准备数据
    folds = [r['fold'] for r in fold_results]
    pccs = [r['pcc'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    mses = [r['mse'] for r in fold_results]

    # 绘制PCC柱状图
    ax1 = plt.subplot(1, 3, 1)
    bars = ax1.bar(folds, pccs, color='skyblue')
    ax1.axhline(y=avg_metrics['pcc'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["pcc"]:.4f} ± {avg_metrics["pcc_std"]:.4f}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('PCC')
    ax1.set_title('PCC by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 在每个柱子上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{pccs[i]:.4f}', ha='center', va='bottom')

    # 绘制MAE柱状图
    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.bar(folds, maes, color='lightgreen')
    ax2.axhline(y=avg_metrics['mae'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["mae"]:.4f} ± {avg_metrics["mae_std"]:.4f}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE by Fold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 在每个柱子上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{maes[i]:.4f}', ha='center', va='bottom')

    # 绘制MSE柱状图
    ax3 = plt.subplot(1, 3, 3)
    bars = ax3.bar(folds, mses, color='salmon')
    ax3.axhline(y=avg_metrics['mse'], color='red', linestyle='--',
                label=f'Avg: {avg_metrics["mse"]:.4f} ± {avg_metrics["mse_std"]:.4f}')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE by Fold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 在每个柱子上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{mses[i]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_results.png'), dpi=300)
    plt.close()

    # 2. 使用高质量绘图函数生成真实值vs预测值图
    # 首先需要收集所有fold的预测和真实值
    all_true = []
    all_pred = []

    # 这里假设可以从fold结果中获取预测值和真实值
    # 如果实际情况不是这样，需要根据情况调整
    for fold_idx, fold_result in enumerate(fold_results):
        # 假设fold_result包含预测值和真实值，或者我们可以加载它们
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        predictions_file = os.path.join(fold_dir, "predictions.npy")

        # 如果存在预测文件，加载并添加到汇总列表
        if os.path.exists(predictions_file):
            fold_data = np.load(predictions_file, allow_pickle=True).item()
            all_true.extend(fold_data['true_values'])
            all_pred.extend(fold_data['predictions'])

    # 如果有收集到预测和真实值，生成高质量图表
    if all_true and all_pred:
        metrics = {
            'PCC': avg_metrics['pcc'],
            'RMSE': np.sqrt(avg_metrics['mse'])  # 假设mse是均方误差
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
    # 计算kde
    kde_points = np.linspace(plot_range[0], plot_range[1], 200)
    kde = gaussian_kde(data)
    kde_values = kde(kde_points)

    # 计算直方图
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 获取暗色版本作为边框色
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
    """Using direct axes management approach with fixed layout issues

    Args:
        true_ddg: 真实值
        predictions: 预测值
        save_path: 保存路径
        metrics: 性能指标字典，包含'PCC', 'RMSE'等
        dataset: 数据集名称
        title: 图表标题
        theme_color: 主题颜色，可以是'red', 'blue', 'purple', 'orange'等有效颜色
    """
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

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(9, 9))

    main_ax = fig.add_axes((0.15, 0.15, 0.7, 0.7))
    top_ax = fig.add_axes((0.15, 0.85, 0.7, 0.15))
    right_ax = fig.add_axes((0.85, 0.15, 0.15, 0.7))

    plot_range = [-10, 10]

    # 计算统计指标
    if metrics:
        pcc = metrics.get('PCC', 0)
        rmse = metrics.get('RMSE', 0)
    else:
        pcc = np.corrcoef(true_ddg, predictions)[0, 1]
        rmse = np.sqrt(np.mean((np.array(true_ddg) - np.array(predictions)) ** 2))

    # 计算拟合直线
    slope, intercept = np.polyfit(true_ddg, predictions, 1)
    # data_range_min = np.percentile(true_ddg, 0.2)
    # data_range_max = np.percentile(true_ddg, 99.8)
    # fit_x = np.array([data_range_min, data_range_max])
    fit_x = np.array([-10, 10])
    fit_line = slope * fit_x + intercept

    # 设置显示范围和刻度
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

    # 绘制散点和直线
    main_ax.scatter(true_ddg, predictions, alpha=0.6, color=theme_color, s=50, zorder=3)
    # main_ax.plot(plot_range, plot_range, '--', color='black', alpha=0.8, linewidth=1.5, zorder=2)
    main_ax.plot(fit_x, fit_line, '-', color=theme_color, alpha=0.8, linewidth=2.0, zorder=2)

    # 直方图
    bins = np.arange(-10, 11, 1)
    plot_enhanced_histogram_v2(top_ax, true_ddg, bins, plot_range, vertical=True, theme_color=theme_color)
    plot_enhanced_histogram_v2(right_ax, predictions, bins, plot_range, vertical=False, theme_color=theme_color)

    # 隐藏直方图刻度
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    right_ax.set_xticks([])
    right_ax.set_yticks([])

    # 控制边框显示
    main_ax.spines['top'].set_visible(False)
    main_ax.spines['right'].set_visible(False)
    top_ax.spines['top'].set_visible(False)
    top_ax.spines['right'].set_visible(False)
    top_ax.spines['left'].set_visible(False)
    right_ax.spines['top'].set_visible(False)
    right_ax.spines['right'].set_visible(False)
    right_ax.spines['bottom'].set_visible(False)

    # 准备文本
    dataset_name = f"{dataset.upper()}" if dataset else ""
    dataset_text = f"{dataset_name}\n"
    metrics_text = f"RMSE: {rmse:.3f}\nPCC   : {pcc:.3f}"
    # 处理截距的符号
    intercept_str = f"- {abs(intercept):.3f}" if intercept < 0 else f"+ {intercept:.3f}"
    equation_text = f"y = {slope:.3f}x {intercept_str}"

    # 添加文本标注
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

    # 添加标签
    main_ax.set_xlabel('True', fontsize=20, fontweight='bold')
    main_ax.set_ylabel('Prediction', fontsize=20, fontweight='bold')

    # 保存图形
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


# ============= 实验相关功能 =============

def extract_metrics_from_directory(directory):
    """从实验目录中提取评估指标"""
    try:
        # 检查best_metrics.json
        metrics_path = os.path.join(directory, "best_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                return metrics.get('test', {})

        # 备选：尝试从test_results.txt中提取
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
        print(f"提取指标出错: {str(e)}")

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
    """可视化PDB限制实验结果"""
    if not results:
        print("没有可用结果进行可视化")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 添加RMSE列 - 计算每个单独结果的RMSE
    df['rmse'] = np.sqrt(df['mse'])

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建plots目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 保存原始结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(output_dir, f"pdb_limited_raw_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"原始结果已保存至: {csv_path}")

    # 计算每个模型+特征类型组合的平均性能和标准差
    summary = df.groupby(['model', 'feature_type', 'feature_name']).agg({
        'pcc': ['mean', 'std', 'count'],
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'rmse': ['mean', 'std']  # 添加RMSE的统计
    }).reset_index()

    # 重命名列以便更易读
    summary.columns = ['model', 'feature_type', 'feature_name',
                       'pcc_mean', 'pcc_std', 'run_count',
                       'mse_mean', 'mse_std',
                       'mae_mean', 'mae_std',
                       'rmse_mean', 'rmse_std']  # 添加RMSE相关列

    # 保存汇总结果
    summary_path = os.path.join(output_dir, f"pdb_limited_summary_{timestamp}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"汇总结果已保存至: {summary_path}")

    # 打印性能汇总
    print("\nPDB限制实验性能汇总 (PCC):")
    for _, row in summary.iterrows():
        print(
            f"{row['model']} + {row['feature_name']}: {row['pcc_mean']:.4f} ± {row['pcc_std']:.4f} (运行 {row['run_count']} 次)")

    # 使用新的分组柱状图替代原来的热图
    comparison_path = os.path.join(plots_dir, f"pdb_limited_comparison_{timestamp}.png")
    create_pdb_limited_comparison_chart(
        data=summary,
        save_path=comparison_path,
        title=None,  # 不使用标题
        y_min=0.5,  # 可自定义y轴范围
        y_max=0.8,
        figsize=(16, 15),
    )
    print(f"高质量PDB限制实验比较图已保存至: {comparison_path}")

    # 创建并保存箱线图 - 使用现有的函数
    boxplot_path = os.path.join(plots_dir, f"pdb_limited_boxplot_{timestamp}.png")
    create_publication_boxplot(
        data=df,
        x='model',
        y='pcc',
        hue='feature_name',
        save_path=boxplot_path,
        # title='Performance Distribution Across Models',
        showpoints=True,
        y_min=0.5,  # 可选：设置Y轴范围
        y_max=0.8,
        figsize=(16, 15),
    )
    print(f"性能分布箱线图已保存至: {boxplot_path}")

    return summary

def create_pdb_limited_comparison_chart(data, save_path, title=None, figsize=(16, 15), y_min=0.6, y_max=None):
    """
    创建针对PDB限制实验的高质量分组柱状图，适合高影响力学术期刊发表

    参数:
        data: 包含结果的DataFrame (必须包含'model', 'feature_name', 'pcc_mean', 'pcc_std'列)
        save_path: 图像保存路径
        title: 图表标题
        figsize: 图像尺寸
        y_min: y轴最小值，默认0.6
        y_max: y轴最大值，默认为None（自动计算）
    """
    # 设置风格以匹配高影响力期刊 - 增大字体
    plt.style.use('default')
    line_width = 1.6
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 28,  # 从22增加到28
        'axes.linewidth': line_width,
        'axes.labelsize': 28,  # 从22增加到28
        'axes.titlesize': 28,  # 从22增加到28
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
        'xtick.labelsize': 30,  # 从24增加到30
        'ytick.labelsize': 30,  # 从24增加到30
        'font.weight': 'normal',
    })

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # 内部实现模型名称映射，可以在这里调整
    model_name_map = {
        'simplified': 'DualSSD',
        'ssd': 'DualSSD*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN'
    }

    # 建议修改
    feature_name_map = {
        'Distribution only': 'Distribution',  # 更简洁
        'No features': 'Baseline',  # 更简洁
        'Intensity only': 'Intensity',  # 更一致
        'Full features': 'Both'  # 更一致
    }

    # 替换特征名称和模型名称
    data_plot = data.copy()
    data_plot['feature_display'] = data_plot['feature_name'].map(lambda x: feature_name_map.get(x, x))

    # 按性能排序模型（降序）
    best_per_model = data_plot.groupby('model')['pcc_mean'].max().reset_index()
    best_per_model = best_per_model.sort_values('pcc_mean', ascending=False)
    sorted_models = best_per_model['model'].tolist()

    # 更新模型名称显示 - 使用映射
    model_display_names = [model_name_map.get(m, m) for m in sorted_models]

    # 过滤并排序数据
    data_plot['model'] = pd.Categorical(data_plot['model'], categories=sorted_models, ordered=True)
    data_plot = data_plot.sort_values('model')

    # 获取特征名称和模型
    feature_names = ['Baseline', 'Distribution', 'Intensity', 'Both']  # 固定顺序
    # 确保只使用数据中实际存在的特征
    feature_names = [f for f in feature_names if f in data_plot['feature_display'].unique()]
    models = data_plot['model'].unique()

    # 设置颜色 - 使用浅色系列配色方案，论文风格
    colors = ['#2E86AB', '#F18F01']  # 蓝橙组合，与箱线图一致

    # 设置柱状图定位
    n_features = len(feature_names)
    bar_width = 0.70 / n_features  # 保持原有的窄柱子
    spacing_factor = 0.9
    positions = np.arange(len(models)) * spacing_factor   # 组间距稍小

    # 先设置y轴范围，这样柱子就会从这个范围开始
    max_val = data_plot['pcc_mean'].max()
    if y_max is None:
        y_max = min(1.0, max_val * 1.05)  # 顶部留出5%的空间
    ax.set_ylim(y_min, y_max)

    # 只保留左边和底部的轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 增强底部边框的视觉重量 - 高级期刊风格
    ax.spines['bottom'].set_linewidth(line_width * 1.1)  # 与箱线图一致
    ax.spines['left'].set_linewidth(line_width * 1.1)  # 与箱线图一致

    # 设置轴线层次 - 确保在柱状图之上
    ax.spines['bottom'].set_zorder(5)
    ax.spines['left'].set_zorder(5)

    # 移除默认网格线
    ax.grid(False)

    # 绘制分组柱状图
    bars_dict = {}
    for i, feature in enumerate(feature_names):
        feature_data = data_plot[data_plot['feature_display'] == feature]
        feature_data = feature_data.set_index('model').reindex(models)

        # 计算偏移量
        offset = (i - n_features / 2 + 0.5) * bar_width
        x_pos = positions + offset

        # 绘制柱状图 - 边缘粗细与坐标轴保持一致
        bars = ax.bar(x_pos, feature_data['pcc_mean'],
                      width=bar_width,
                      color=colors[i],
                      edgecolor='black',
                      linewidth=line_width * 0.9,  # 与箱线图的线宽计算方式一致
                      alpha=0.92,
                      label=feature,
                      zorder=3)
        bars_dict[feature] = bars

        # 添加误差线 - 保持当前粗细，符合高级期刊美学
        if 'pcc_std' in feature_data.columns:
            ax.errorbar(x_pos, feature_data['pcc_mean'],
                        yerr=feature_data['pcc_std'],
                        fmt='none',
                        ecolor='black',
                        elinewidth=1.2,    # 稍微加粗，与整体线宽协调
                        capsize=4,
                        zorder=4)

        # 将此代码替换柱状图中原有的文本设置部分
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):  # 只在有值时添加标签
                # 获取柱子的颜色
                bar_color = bar.get_facecolor()

                # 计算颜色的亮度 (使用感知亮度公式)
                # 这个公式考虑了人眼对不同颜色亮度的感知差异
                r, g, b, a = bar_color
                brightness = 0.299 * r + 0.587 * g + 0.114 * b

                # 根据亮度选择文本颜色
                text_color = 'white' if brightness < 0.55 else 'black'

                # 获取柱子的实际底部位置
                bar_bottom = y_min  # 柱子始终从y_min开始

                # 设置文本位置为柱子底部上方一点点
                text_offset = 0.008  # 从底部稍微偏移一点点
                y_pos = bar_bottom + text_offset

                # 添加动态颜色的文本
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        f'{height:.3f}',  # 改为3位小数
                        ha='center',
                        va='bottom',
                        fontsize=24,  # 从20增加到24
                        fontweight='normal',  # 与箱线图文字粗细一致
                        color=text_color,
                        rotation=90,
                        # 添加与箱线图一致的描边效果
                        path_effects=[path_effects.Stroke(linewidth=0.2, foreground=text_color),
                                      path_effects.Normal()],
                        zorder=5)

    # 配置坐标轴 - 使用标准方法设置标签
    ax.set_xticks(positions)
    # 在设置X轴刻度后添加：

    ax.set_xlim(-0.5, len(models) * spacing_factor - 0.5)  # spacing_factor是你用的倍数
    ax.set_xticklabels(model_display_names, rotation=40, ha='right')

    # 坐标轴的刻度放在柱状图上层, 确保刻度标签使用正常字重
    ax.tick_params(axis='both', which='both', zorder=6)

    # 移除x轴标签 - 符合顶级期刊简洁风格
    ax.set_xlabel('')  # 删除"Model"标签

    # 仅保留Y轴标签
    ax.set_ylabel('PCC', fontsize=28, fontweight='normal', labelpad=18)  # 从22增加到28

    # 如果使用标题，使用更专业的字重
    if title:
        ax.set_title('Model Performance with Different Features', fontsize=30, fontweight='normal', pad=12)  # 从24增加到30

    # 添加图例 - 增大字体使其更醒目
    legend = ax.legend(
        frameon=True,
        framealpha=0.75,  # 增加透明度
        edgecolor='#555555',  # 灰色边框更柔和
        fontsize=26,  # 从22增加到26
        loc='upper right'
    )

    # 更好的留白
    plt.tight_layout(pad=1.5)

    # 保存高质量图像
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
        'simplified': 'DualSSD',
        'ssd': 'DualSSD*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN'
    }
    # 特征名称映射
    feature_name_map = {
        'Distribution only': 'Distribution',
        'No features': 'Baseline',
        'Intensity only': 'Intensity',
        'Full features': 'Both'
    }

    # 应用映射到数据
    data_mapped = data.copy()

    # 如果x列是模型，则映射模型名称
    if x in data_mapped.columns:
        data_mapped[x] = data_mapped[x].map(lambda x_val: model_name_map.get(x_val, x_val))

    # 如果hue列是特征，则映射特征名称
    if hue in data_mapped.columns:
        data_mapped[hue] = data_mapped[hue].map(lambda hue_val: feature_name_map.get(hue_val, hue_val))

        # ===== 在这里插入排序代码 =====
        # 计算每个模型在"Base"条件下的中位数并排序
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

        # 按中位数降序排序
        base_medians.sort(key=lambda x: x[1], reverse=True) # 如果想要升序排序,改为False
        sorted_models = [model for model, _ in base_medians]

        print("模型按Base中位数排序:")
        for model, median in base_medians:
            print(f"  {model}: {median:.3f}")

        # 应用排序
        data_mapped[x] = pd.Categorical(data_mapped[x], categories=sorted_models, ordered=True)

        # 添加以下代码：
        # 固定特征顺序，确保与柱状图一致
        fixed_feature_order = ['Baseline', 'Distribution', 'Intensity', 'Both']
        existing_features = [f for f in fixed_feature_order if f in data_mapped[hue].unique()]
        data_mapped[hue] = pd.Categorical(data_mapped[hue], categories=existing_features, ordered=True)
        data_mapped = data_mapped.sort_values([x, hue])  # 重新排序
        # ===== 排序代码结束 =====

    # Set style to match top-tier journals - 增大字体
    plt.style.use('default')
    line_width = 1.6
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 28,  # 从22增加到28
        'axes.linewidth': line_width,
        'axes.labelsize': 28,  # 从22增加到28
        'axes.titlesize': 28,  # 从22增加到28
        'xtick.major.width': line_width,
        'ytick.major.width': line_width,
        'xtick.major.size': 10,
        'ytick.major.size': 10,
        'xtick.labelsize': 30,  # 从24增加到30
        'ytick.labelsize': 30,  # 从24增加到30
        'font.weight': 'normal',  # 改为正常字重
    })

    # Set default palette if none provided
    if palette is None:
        palette = ['#2E86AB', '#F18F01']  # 深蓝 + 亮橙
        palette = sns.color_palette(palette,  len(data[hue].unique()))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Calculate data range for y-axis limits
    if y_min is None or y_max is None:
        y_data_min = data_mapped[y].min()  # 改这里
        y_data_max = data_mapped[y].max()  # 改这里

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
        fliersize=5 if showfliers else 0,  # 减小异常值点大小
        linewidth=line_width * 0.9,  # 箱体线条稍细，更精致
        whiskerprops=dict(linewidth=line_width * 1.2, color='black'),  # 修改：T线改为纯黑色，稍微加粗
        medianprops=dict(color='black', linewidth=line_width * 1.4),
        capprops=dict(linewidth=line_width * 1.2, color='black'),  # 修改：cap线也改为黑色
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
            size=5,  # 减小点大小
            alpha=0.6,  # 增加透明度
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

        # 将文字放在箱子中心稍下的位置
        text_y_pos = median - 0.02 * (y_max - y_min)  # 稍微在中位线上方

        # === 新增：获取箱体颜色并计算亮度 ===
        # 从调色板中获取当前hue对应的颜色
        color = palette[hue_index]
        if isinstance(color, str):
            # 如果是字符串颜色，需要转换为RGB
            import matplotlib.colors as mcolors
            color = mcolors.to_rgba(color)

        # 计算颜色的亮度
        r, g, b = color[:3]  # 取RGB值
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # 根据亮度选择文本颜色
        text_color = 'white' if brightness < 0.55 else 'black'
        # === 新增部分结束 ===

        ax.text(
            x_index + offset,
            text_y_pos,
            f"{median:.2f}",
            ha='center',
            va='center',
            fontsize=17,  # 保持原有大小，因为用户说可能无法调整
            fontweight='normal',
            color=text_color,  # 使用动态计算的颜色
            path_effects=[path_effects.Stroke(linewidth=0.2, foreground=text_color),
                          path_effects.Normal()],
            zorder=6,
        )

    # Refine plot styling - enhance bottom border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(line_width * 1.1)  # 轻微加粗
    ax.spines['left'].set_linewidth(line_width * 1.1)  # 轻微加粗

    # Set layer order
    ax.spines['bottom'].set_zorder(5)
    ax.spines['left'].set_zorder(5)
    ax.tick_params(axis='both', which='both', zorder=6)

    # Enhance axis labels with normal weight
    ax.set_xlabel('', fontsize=28, fontweight='normal', labelpad=18)  # 从22增加到28
    ax.set_ylabel('PCC', fontsize=28, fontweight='normal', labelpad=18)  # 从22增加到28
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right')

    # Add title if provided - with refined styling
    if title:
        box_style = dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.9, edgecolor='#e0e0e0')  # 更淡的边框
        ax.set_title(title, fontsize=30, fontweight='normal', pad=20, bbox=box_style)  # 从24增加到30

    # Create more elegant legend
    legend = ax.legend(
        frameon=True,
        framealpha=0.8,  # 与箱线图一致
        edgecolor='#555555',  # 与箱线图一致
        fontsize=18,  # 从18增加到22
        title_fontsize=24,  # 从20增加到24
        loc='upper right'
    )

    # 确保图例标题使用正常字重
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
    """
    创建发表级别的热图，优化行高和颜色条，支持自定义Y轴顺序

    参数:
        data: DataFrame - 数据矩阵
        save_path: str - 保存路径
        title: str - 图表标题
        figsize: tuple - 图像尺寸
        cmap: str - 颜色映射
        annot_fmt: str - 注释格式
        vmin, vmax: float - 颜色范围
        custom_order: list - 自定义Y轴模型顺序，如果为None则使用默认顺序
    """

    # 模型名称映射
    name_mapping = {
        'simplified': 'DualSSD',
        'ssd': 'DualSSD*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN',
    }

    # 特征类型映射（同时定义顺序）
    feature_mapping = {
        'No features': 'Baseline',  # 基线在前
        'Distribution only': 'Distribution',
        'Intensity only': 'Intensity',
        'Full features': 'All',  # 完整特征在最后
    }

    # 新增：阈值集合名称映射（同时定义顺序）
    threshold_mapping = {
        'Baseline': 'Baseline',        # 1. 基准对比
        'fine': 'Fine-grained',        # 3. 最精细粒度 [6.5-19.0]
        'near': 'Near-range',          # 4. 近程相互作用 [6.0-20.0]
        'dense': 'Dense',              # 5. 密集采样 [7.0-22.0]
        'hydrophobic': 'Hydrophobic',  # 6. 疏水相互作用 [7.0-25.0]
        'default': 'Contact',  # 2. 标准配置Balanced（技术）或Contact（生物）比较合适
        'electrostatic': 'Electrostatic', # 7. 静电效应 [8.0-65.0]
        'domain': 'Domain',            # 8. 结构域耦合 [10.0-85.0]
        'sparse': 'Sparse',            # 9. 稀疏采样 [8.0-90.0]
        'coarse': 'Coarse-grained'     # 10. 最粗粒度 [10.0-120.0]
    }

    # === 打印调试信息 ===
    print("原始数据索引:", data.index.tolist())
    print("原始数据列:", data.columns.tolist())

    # 应用名称映射
    data_mapped = data.copy()
    data_mapped.index = [name_mapping.get(idx, idx) for idx in data.index]

    print("映射后的索引:", data_mapped.index.tolist())
    print("原始列:", data_mapped.columns.tolist())

    # 🔥 检查数据类型并应用相应的映射和排序
    has_feature_data = any(col in feature_mapping for col in data_mapped.columns)
    has_threshold_data = any(col in threshold_mapping for col in data_mapped.columns)

    if has_feature_data:
        # 应用特征映射到列名
        data_mapped.columns = [feature_mapping.get(col, col) for col in data_mapped.columns]

        # 根据feature_mapping的顺序重新排列列
        available_columns = data_mapped.columns.tolist()
        feature_order = list(feature_mapping.values())  # 使用映射后的值作为顺序

        # 保留在数据中存在且在feature_mapping中定义的列，按feature_mapping顺序排列
        ordered_columns = [col for col in feature_order if col in available_columns]
        # 添加任何不在feature_mapping中但存在于数据中的列
        missing_columns = [col for col in available_columns if col not in ordered_columns]
        final_column_order = ordered_columns + missing_columns

        print("检测到特征数据，应用特征映射和排序")
        print("最终列顺序:", final_column_order)
        data_mapped = data_mapped.reindex(columns=final_column_order)

    elif has_threshold_data:
        # 应用阈值映射到列名
        data_mapped.columns = [threshold_mapping.get(col, col) for col in data_mapped.columns]

        # 根据threshold_mapping的顺序重新排列列
        available_columns = data_mapped.columns.tolist()
        threshold_order = list(threshold_mapping.values())  # 使用映射后的值作为顺序

        # 保留在数据中存在且在threshold_mapping中定义的列，按threshold_mapping顺序排列
        ordered_columns = [col for col in threshold_order if col in available_columns]
        # 添加任何不在threshold_mapping中但存在于数据中的列
        missing_columns = [col for col in available_columns if col not in ordered_columns]
        final_column_order = ordered_columns + missing_columns

        print("检测到阈值数据，应用阈值映射和排序")
        print("最终列顺序:", final_column_order)
        data_mapped = data_mapped.reindex(columns=final_column_order)

    else:
        # 既不是特征数据也不是阈值数据，保持原顺序
        print("未检测到特征或阈值数据，使用原始列顺序")
        print("最终列顺序:", data_mapped.columns.tolist())

    # 更安全的方法：不使用loc，而是重新排序索引
    if custom_order is not None:
        print("尝试使用自定义行顺序:", custom_order)

        # 1. 找出原始数据中存在的模型
        available_models = data_mapped.index.tolist()
        print("可用的模型:", available_models)

        # 2. 保留自定义顺序中在原始数据中存在的模型
        valid_order = [model for model in custom_order if model in available_models]
        print("有效的自定义顺序:", valid_order)

        # 3. 添加任何不在自定义顺序中但在原始数据中的模型（放在最后）
        missing_models = [model for model in available_models if model not in valid_order]
        final_order = valid_order + missing_models
        print("最终使用的行顺序:", final_order)

        # 4. 使用reindex而不是loc来排序
        data_mapped = data_mapped.reindex(final_order)

    # 设置风格
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif'],
        'font.size': 24,
        'axes.linewidth': 1.2,
    })

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # 绘制热图
    hm = sns.heatmap(
        data_mapped,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        linewidths=1,
        linecolor='white',
        cbar_kws={
            'shrink': 1.0,  # 使颜色条与主图高度匹配
            'aspect': aspect,  # 控制颜色条的宽度
            'pad': 0.02,  # 控制颜色条与主图的间距
            'ticks': [0.8, 0.85, 0.9, 0.95],  # 明确指定刻度位置
        },
        square=False,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    # 改进的文本颜色设置 - 基于背景亮度
    cmap_obj = plt.cm.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)

    for i, j in np.ndindex(data_mapped.shape):
        if j < len(data_mapped.columns) and i < len(data_mapped.index):
            # 获取当前单元格的值
            try:
                value = data_mapped.iloc[i, j]
                if not np.isnan(value):
                    # 获取对应的颜色
                    rgba = cmap_obj(norm(value))
                    # 计算亮度 (基于RGB值的加权平均，接近人眼感知)
                    brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

                    # 获取文本对象
                    idx = i * len(data_mapped.columns) + j
                    if idx < len(ax.texts):
                        text = ax.texts[idx]
                        # 根据亮度设置文本颜色
                        if brightness < 0.50:  # 阈值可以调整
                            text.set_color('white')
                        else:
                            text.set_color('black')

                        # 设置字体大小
                        text.set_fontsize(20)
            except (IndexError, ValueError):
                pass

    # # 设置标题
    # if title is None:
    #     title = "PCC performance of different models in 5-fold cross-validation"
    ax.set_title(title, fontsize=24, fontweight='normal', pad=15)

    # 调整刻度标签
    plt.setp(ax.get_xticklabels(),
             rotation=40, ha='right',
             fontsize=20, fontweight='normal')
    plt.setp(ax.get_yticklabels(),
             rotation=0,
             fontsize=20, fontweight='normal')

    # 获取颜色条并设置刻度标签格式
    cbar = ax.collections[0].colorbar
    # cbar.ax.set_yticklabels([f"{x:.1f}" for x in cbar.get_ticks()])  # 一位小数
    # 设置颜色条刻度标签字体大小
    # 或者更详细的设置：
    cbar.ax.tick_params(
        labelsize=20,  # 字体大小
        colors='black',  # 字体颜色
        width=1.2,  # 刻度线宽度
        length=4  # 刻度线长度
    )

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)

    print(f"热图已保存至: {save_path}")
    return fig, ax

def visualize_results(results, output_dir):
    """
    使用高质量热图可视化实验矩阵结果
    """
    if not results:
        print("没有可用结果进行可视化")
        return

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 创建交叉表以便可视化
    pcc_pivot = pd.pivot_table(df, values='pcc', index=['model'], columns=['feature_name'])

    # 获取时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存表格数据
    df.to_csv(os.path.join(output_dir, f"experiment_matrix_results_{timestamp}.csv"), index=False)

    # 创建论文级热图
    heatmap_path = os.path.join(output_dir, f"publication_heatmap_{timestamp}.png")
    create_publication_heatmap(
        data=pcc_pivot,
        save_path=heatmap_path,
        title="PCC values of all methods on RNA-binding prediction",
        # 可选: 设置特定的值范围
        vmin=0.75,  # 调整这些值以获得更好的色彩对比
        vmax=0.95
    )

    # 打印性能汇总
    print("\n实验矩阵性能汇总 (PCC):")
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
        'simplified': 'DualSSD',
        'ssd': 'DualSSD*',
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
    Create an enhanced publication-quality plot with DualSSD prominently highlighted

    Parameters:
        results_df: DataFrame with columns ['model', 'train_ratio', 'pcc']
        output_path: Path to save the figure
        figsize: Figure size tuple
        y_min, y_max: Y-axis limits
        manual_offsets: Dict mapping train_ratio to offset value for DualSSD annotations
                       e.g., {10: 25, 20: 15, 30: 20, 40: 18, 50: 22, 60: 35, 70: 15, 80: 18, 90: 15}
                       If None, uses adaptive positioning
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib as mpl
    from matplotlib import patheffects

    # Set style for refined scientific plots - 增大字体
    plt.style.use('default')
    line_width = 1.8  # Increased base line width

    plt.rcParams.update({
        'font.family': ['serif'],
        'font.serif': ['DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 24,  # 从18增加到24
        'axes.linewidth': 1.2,
        'axes.labelsize': 28,  # 从22增加到28
        'axes.titlesize': 30,  # 从24增加到30
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.labelsize': 26,  # 从20增加到26
        'ytick.labelsize': 26,  # 从20增加到26
    })

    # Create figure with enhanced size
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi=150)
    ax.set_facecolor('white')

    # Get unique models
    models = results_df['model'].unique()

    # INTERNAL NAME MAPPING
    name_mapping = {
        'simplified': 'DualSSD',
        'ssd': 'DualSSD*',
        'transformer': 'Transformer',
        'graph': 'GraphTrans',
        'gcn': 'GCN',
        'gat': 'GAT',
        'gin': 'GIN',
        'sage': 'GraphSAGE',
        'edge': 'DGCNN',
        'mamba_triple': 'Mamba',
    }

    # Enhanced marker mapping with more distinctive shapes
    marker_map = {
        'simplified': 'o',  # Circle for DualSSD
        'ssd': 's',  # Square
        'transformer': '^',  # Triangle up
        'graph': 'D',  # Diamond
        'gcn': 'v',  # Triangle down
        'gat': 'p',  # Pentagon
        'gin': '*',  # Star
        'sage': 'h',  # Hexagon
        'edge': 'X',  # X
        'mamba_triple': 'P',  # Plus
    }

    # Enhanced color scheme - DualSSD gets the most striking color
    color_map = {
        'simplified': '#E63946',  # Bright red for DualSSD - most eye-catching
        'ssd': '#457B9D',  # Steel blue
        'transformer': '#F77F00',  # Orange
        'graph': '#6A994E',  # Green
        'gcn': '#A663CC',  # Purple
        'gat': '#F72585',  # Pink
        'gin': '#4CC9F0',  # Light blue
        'sage': '#7209B7',  # Dark purple
        'edge': '#FB8500',  # Dark orange
        'mamba_triple': '#219EBC',  # Teal
    }

    # Calculate model performance for sorting - use mean PCC instead of max
    model_performance = {}
    for model in models:
        model_data = results_df[results_df['model'] == model]
        if not model_data.empty:
            mean_pcc = model_data['pcc'].mean()  # 使用平均值而不是最大值
            model_performance[model] = mean_pcc

    # Sort models by performance (descending)
    sorted_models = sorted(models, key=lambda m: model_performance.get(m, 0), reverse=True)

    # Add subtle background gradient for visual appeal
    ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.02, color='lightblue', zorder=0)

    # Plot each model
    lines = []
    labels = []
    model_line_map = {}  # 添加映射来跟踪模型和线条的对应关系

    for i, model in enumerate(sorted_models):
        model_data = results_df[results_df['model'] == model].sort_values('train_ratio')

        marker = marker_map.get(model, 'o')
        color = color_map.get(model, '#666666')
        display_name = name_mapping.get(model, model)

        # Special treatment for DualSSD (our best method)
        if model == 'simplified':  # DualSSD
            # First, plot a glow effect (wider, transparent line)
            ax.plot(
                model_data['train_ratio'] * 100,
                model_data['pcc'],
                color=color,
                marker=marker,
                markersize=14,
                markeredgecolor=color,
                markeredgewidth=0,
                linewidth=8,  # Much thicker for glow
                alpha=0.3,  # Transparent for glow effect
                zorder=8,
            )

            # Then plot the main line on top
            line, = ax.plot(
                model_data['train_ratio'] * 100,
                model_data['pcc'],
                label=display_name,
                color=color,
                marker=marker,
                markersize=12,  # Larger markers
                markeredgecolor='white',
                markeredgewidth=2.5,  # Thicker edge
                linewidth=4.5,  # Much thicker line
                alpha=0.95,
                zorder=10,  # On top of everything
            )

        else:
            # Regular plotting for other methods
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
                alpha=0.85,
                zorder=5 + (len(models) - i)
            )

        lines.append(line)
        labels.append(display_name)
        model_line_map[model] = line  # 保存映射关系

        # Add value annotations for the best method at all points
        if model == 'simplified':
            # Annotate all points for DualSSD with adaptive or manual positioning
            for _, row in model_data.iterrows():
                train_ratio = row['train_ratio']
                dualssd_pcc = row['pcc']

                # Determine offset: manual if provided, otherwise adaptive
                if manual_offsets and (train_ratio * 100) in manual_offsets:
                    # Use manual offset for this training ratio
                    y_offset = manual_offsets[train_ratio * 100]
                else:
                    # Use adaptive positioning
                    same_ratio_data = results_df[results_df['train_ratio'] == train_ratio]
                    max_pcc_at_ratio = same_ratio_data['pcc'].max()

                    # Calculate adaptive offset: place annotation above the highest method at this ratio
                    base_offset = 12
                    y_offset = base_offset + (max_pcc_at_ratio - dualssd_pcc) * 1000

                ax.annotate(
                    f'{dualssd_pcc:.3f}',
                    xy=(train_ratio * 100, dualssd_pcc),
                    xytext=(0, y_offset),  # 使用计算出的偏移量
                    textcoords='offset points',
                    fontsize=22,  # 从16增加到22
                    fontweight='normal',
                    ha='center',
                    va='bottom',
                    color='black',
                    zorder=15
                )

    # Enhanced axis configuration
    ax.set_xlabel('Training Set Ratio (%)', fontweight='normal', fontsize=28)  # 从22增加到28
    ax.set_ylabel('PCC', fontweight='normal', fontsize=28)  # 从22增加到28

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
    ax.spines['bottom'].set_linewidth(1.3)  # 更细的坐标轴
    ax.spines['left'].set_linewidth(1.3)  # 更细的坐标轴

    # Enhanced legend with special treatment for DualSSD and performance order
    # Create legend entries in performance order
    legend_elements = []
    legend_labels = []

    for model in sorted_models:  # Already sorted by performance
        line = model_line_map[model]  # 使用映射获取对应的线条
        display_name = name_mapping.get(model, model)

        if model == 'simplified':  # DualSSD
            # Special legend entry for DualSSD
            legend_elements.append(
                plt.Line2D([0], [0],
                           color=line.get_color(),
                           marker=line.get_marker(),
                           markersize=10,
                           markeredgecolor='white',
                           markeredgewidth=2,
                           linewidth=4,
                           label=f'{display_name} (Best)',
                           alpha=0.95)
            )
            legend_labels.append(f'{display_name} (Best)')
        else:
            legend_elements.append(line)
            legend_labels.append(display_name)

    legend = ax.legend(
        legend_elements, legend_labels,
        loc='lower right',
        bbox_to_anchor=(0.98, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fontsize=18,  # 保持原有大小 - 用户要求右下角不调大
        handlelength=2.0,
        columnspacing=1.5,
        labelspacing=0.8,
        title='Methods',
        title_fontsize=18,  # 保持原有大小 - 用户要求右下角不调大
    )

    # Make DualSSD legend text bold
    for text in legend.get_texts():
        if 'Best' in text.get_text():
            text.set_fontweight('bold')
            text.set_color('#E63946')  # Same as line color

    # Add performance ranking annotation
    best_model_name = name_mapping.get(sorted_models[0], sorted_models[0])
    best_avg_pcc = model_performance[sorted_models[0]]
    ax.text(
        0.05, 0.98,
        f'Best: {best_model_name} ({best_avg_pcc:.3f})',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E63946', alpha=0.1, edgecolor='#E63946'),
        fontsize=24,  # 从18增加到24
        fontweight='bold',
        ha='left',
        va='top',
        color='#E63946'
    )

    # Enhanced layout
    plt.tight_layout(pad=1.0)

    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    pdf_path = output_path.replace('.png', '.pdf') if output_path.endswith('.png') else output_path + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    return fig, ax

def plot_training_ratio_example():
    """
    Example function showing how to use plot_training_ratio_curves
    """
    import pandas as pd
    import numpy as np
    import os

    # Create sample data
    models = ['dualssd', 'gat', 'gcn', 'transformer']
    train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Create an empty list to store data rows
    data = []

    # Generate synthetic data for example
    for model in models:
        # Base performance level for each model
        if model == 'dualssd':
            base = 0.65
            slope = 0.15
        elif model == 'gat':
            base = 0.6
            slope = 0.1
        elif model == 'gcn':
            base = 0.58
            slope = 0.12
        else:
            base = 0.55
            slope = 0.08

        for ratio in train_ratios:
            # Calculate PCC with some randomness
            pcc = base + slope * ratio + np.random.normal(0, 0.01)

            # Ensure PCC is in valid range
            pcc = min(1.0, max(0.0, pcc))

            # Add to data
            data.append({
                'model': model,
                'train_ratio': ratio,
                'pcc': pcc
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create output directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)

    # Plot the data
    fig, ax = plot_publication_training_ratio(
        df,
        'examples/training_ratio_example.png',
        # title='Effect of Training Set Size on Model Performance',
        y_min=0.5
    )

    print("Example plot saved to examples/training_ratio_example.png")
    return df


def main():
    """
    Main function to run the example
    """
    plot_training_ratio_example()


if __name__ == '__main__':
    main()
