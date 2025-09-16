import os
import numpy as np
import pickle
import pandas as pd
import networkx as nx
from torch.ao.nn.quantized.functional import threshold
from tqdm import tqdm
from Bio import PDB
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import logging
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("improved_interface_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从现有代码导入
from utils import (create_graph_from_structure, get_residue_ca_coords,
                   positional_encoding, load_conservation_scores, get_rna_center_coords)
from enhanced_rna_processing_debug import create_rna_graph_from_structure


# ==================== 增强的界面检测函数 ====================

def calculate_contact_maps(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    """
    计算不同阈值下的接触图

    参数:
    - protein_coords: 蛋白质坐标
    - rna_coords: RNA坐标
    - thresholds: 距离阈值列表

    返回:
    - 字典，键为阈值，值为接触图矩阵
    """
    contact_maps = {}

    # 计算距离矩阵
    dist_matrix = cdist(protein_coords, rna_coords)

    # 计算不同阈值下的接触图
    for threshold in thresholds:
        contact_maps[threshold] = (dist_matrix < threshold)

    return contact_maps, dist_matrix


def identify_multiscale_interface(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    """
    多尺度界面识别

    参数:
    - protein_coords: 蛋白质坐标
    - rna_coords: RNA坐标
    - thresholds: 距离阈值列表

    返回:
    - interface_info: 包含界面对、掩码和分析信息的字典
    """
    if len(protein_coords) == 0 or len(rna_coords) == 0:
        return {
            'pairs': [],
            'protein_mask': np.zeros(len(protein_coords), dtype=bool),
            'rna_mask': np.zeros(len(rna_coords), dtype=bool),
            'contact_counts': {},
            'contact_maps': {}
        }

    # 计算接触图
    contact_maps, dist_matrix = calculate_contact_maps(protein_coords, rna_coords, thresholds)

    # 找到所有界面对
    all_pairs = []

    for threshold in thresholds:
        contacts = contact_maps[threshold]
        p_indices, r_indices = np.where(contacts)

        for p_idx, r_idx in zip(p_indices, r_indices):
            # 只添加尚未存在的对
            pair_key = (p_idx, r_idx)
            if not any(p['protein_idx'] == p_idx and p['rna_idx'] == r_idx for p in all_pairs):
                all_pairs.append({
                    'protein_idx': p_idx,
                    'rna_idx': r_idx,
                    'distance': dist_matrix[p_idx, r_idx],
                    'threshold': threshold
                })

    # 创建界面掩码
    protein_mask = np.zeros(len(protein_coords), dtype=bool)
    rna_mask = np.zeros(len(rna_coords), dtype=bool)

    for pair in all_pairs:
        protein_mask[pair['protein_idx']] = True
        rna_mask[pair['rna_idx']] = True

    # 统计不同阈值下的接触数量
    contact_counts = {threshold: np.sum(contact_maps[threshold]) for threshold in thresholds}

    return {
        'pairs': all_pairs,
        'protein_mask': protein_mask,
        'rna_mask': rna_mask,
        'contact_counts': contact_counts,
        'contact_maps': contact_maps,
        'dist_matrix': dist_matrix
    }


def create_segmented_contact_features(protein_coords, rna_coords,
                                                thresholds=[8.0, 10.0, 15.0, 20.0, 30.0, 50.0], use_torch=True):
    """
    优化版本：创建分段式多尺度接触特征 (使用分段区间而非累积阈值)

    参数:
    - protein_coords: 蛋白质坐标数组
    - rna_coords: RNA坐标数组
    - thresholds: 距离阈值列表
    - use_torch: 是否使用PyTorch加速计算

    返回:
    - protein_contact_features: 蛋白质节点的接触特征 (分段区间分布)
    - rna_contact_features: RNA节点的接触特征 (分段区间分布)
    - protein_contact_intensity: 蛋白质节点的接触强度
    - rna_contact_intensity: RNA节点的接触强度
    """
    import numpy as np

    # 初始化特征矩阵 (特征维度为阈值数+1，包含最后的>max_threshold区间)
    num_classes = len(thresholds) + 1
    num_protein = len(protein_coords)
    num_rna = len(rna_coords)

    # 如果没有坐标，返回空特征
    if num_protein == 0 or num_rna == 0:
        protein_contact_features = np.zeros((num_protein, num_classes))
        rna_contact_features = np.zeros((num_rna, num_classes))
        protein_contact_intensity = np.zeros((num_protein, 1))
        rna_contact_intensity = np.zeros((num_rna, 1))
        return protein_contact_features, rna_contact_features, protein_contact_intensity, rna_contact_intensity

    # 使用PyTorch加速计算
    if use_torch:
        import torch

        # 转换为tensor
        if not isinstance(protein_coords, torch.Tensor):
            protein_coords_tensor = torch.tensor(protein_coords, dtype=torch.float32)
        else:
            protein_coords_tensor = protein_coords.float()

        if not isinstance(rna_coords, torch.Tensor):
            rna_coords_tensor = torch.tensor(rna_coords, dtype=torch.float32)
        else:
            rna_coords_tensor = rna_coords.float()

        # 计算距离矩阵
        dist_matrix = torch.cdist(protein_coords_tensor, rna_coords_tensor)

        # 初始化特征矩阵
        protein_contact_features = torch.zeros((num_protein, num_classes))
        rna_contact_features = torch.zeros((num_rna, num_classes))

        # 获取每个蛋白质残基的最小距离
        protein_min_dist, _ = torch.min(dist_matrix, dim=1)
        protein_intensity = 1.0 / (1.0 + torch.exp((protein_min_dist - 8.0) / 2.0))
        protein_contact_intensity = protein_intensity.unsqueeze(1)

        # 获取每个RNA核苷酸的最小距离
        rna_min_dist, _ = torch.min(dist_matrix, dim=0)
        rna_intensity = 1.0 / (1.0 + torch.exp((rna_min_dist - 8.0) / 2.0))
        rna_contact_intensity = rna_intensity.unsqueeze(1)

        # 计算蛋白质节点的分段接触特征
        # 第一个区间: < thresholds[0]
        mask = (dist_matrix < thresholds[0])
        protein_contact_features[:, 0] = mask.sum(dim=1).float() / num_rna

        # 中间区间: thresholds[j-1] <= dist < thresholds[j]
        for j in range(1, len(thresholds)):
            mask = (dist_matrix >= thresholds[j - 1]) & (dist_matrix < thresholds[j])
            protein_contact_features[:, j] = mask.sum(dim=1).float() / num_rna

        # 最后一个区间: >= 最大阈值
        mask = (dist_matrix >= thresholds[-1])
        protein_contact_features[:, -1] = mask.sum(dim=1).float() / num_rna

        # 计算RNA节点的分段接触特征
        # 第一个区间: < thresholds[0]
        mask = (dist_matrix < thresholds[0])
        rna_contact_features[:, 0] = mask.sum(dim=0).float() / num_protein

        # 中间区间: thresholds[j-1] <= dist < thresholds[j]
        for j in range(1, len(thresholds)):
            mask = (dist_matrix >= thresholds[j - 1]) & (dist_matrix < thresholds[j])
            rna_contact_features[:, j] = mask.sum(dim=0).float() / num_protein

        # 最后一个区间: >= 最大阈值
        mask = (dist_matrix >= thresholds[-1])
        rna_contact_features[:, -1] = mask.sum(dim=0).float() / num_protein

        # 转换回NumPy数组
        protein_contact_features = protein_contact_features.numpy()
        rna_contact_features = rna_contact_features.numpy()
        protein_contact_intensity = protein_contact_intensity.numpy()
        rna_contact_intensity = rna_contact_intensity.numpy()

    # 使用NumPy计算（较慢）
    else:
        from scipy.spatial.distance import cdist

        # 初始化特征矩阵
        protein_contact_features = np.zeros((num_protein, num_classes))
        rna_contact_features = np.zeros((num_rna, num_classes))
        protein_contact_intensity = np.zeros((num_protein, 1))
        rna_contact_intensity = np.zeros((num_rna, 1))

        # 计算距离矩阵
        dist_matrix = cdist(protein_coords, rna_coords)

        # 为每个蛋白质节点计算接触分布和强度
        for i in range(num_protein):
            distances = dist_matrix[i]

            # 计算接触强度 (基于最小距离)
            min_dist = np.min(distances)
            intensity = 1.0 / (1.0 + np.exp((min_dist - 8.0) / 2.0))
            protein_contact_intensity[i, 0] = intensity

            # 计算分段区间分布
            # 第一个区间: < thresholds[0]
            mask = (distances < thresholds[0])
            protein_contact_features[i, 0] = np.sum(mask) / num_rna

            # 中间区间: thresholds[j-1] <= dist < thresholds[j]
            for j in range(1, len(thresholds)):
                mask = (distances >= thresholds[j - 1]) & (distances < thresholds[j])
                protein_contact_features[i, j] = np.sum(mask) / num_rna

            # 最后一个区间: >= 最大阈值
            mask = (distances >= thresholds[-1])
            protein_contact_features[i, -1] = np.sum(mask) / num_rna

        # 为每个RNA节点计算接触分布和强度
        for i in range(num_rna):
            distances = dist_matrix[:, i]

            # 计算接触强度 (基于最小距离)
            min_dist = np.min(distances)
            intensity = 1.0 / (1.0 + np.exp((min_dist - 8.0) / 2.0))
            rna_contact_intensity[i, 0] = intensity

            # 计算分段区间分布
            # 第一个区间: < thresholds[0]
            mask = (distances < thresholds[0])
            rna_contact_features[i, 0] = np.sum(mask) / num_protein

            # 中间区间: thresholds[j-1] <= dist < thresholds[j]
            for j in range(1, len(thresholds)):
                mask = (distances >= thresholds[j - 1]) & (distances < thresholds[j])
                rna_contact_features[i, j] = np.sum(mask) / num_protein

            # 最后一个区间: >= 最大阈值
            mask = (distances >= thresholds[-1])
            rna_contact_features[i, -1] = np.sum(mask) / num_protein

    return protein_contact_features, rna_contact_features, protein_contact_intensity, rna_contact_intensity


def create_multiscale_contact_features(protein_coords, rna_coords, thresholds=[8.0, 12.0, 16.0]):
    """
    创建多尺度接触特征

    参数:
    - protein_coords: 蛋白质坐标数组
    - rna_coords: RNA坐标数组
    - thresholds: 距离阈值列表

    返回:
    - protein_contact_features: 蛋白质节点的接触特征
    - rna_contact_features: RNA节点的接触特征
    """
    # 初始化特征矩阵
    protein_contact_features = np.zeros((len(protein_coords), len(thresholds)))
    rna_contact_features = np.zeros((len(rna_coords), len(thresholds)))

    # 如果没有坐标，返回空特征
    if len(protein_coords) == 0 or len(rna_coords) == 0:
        return protein_contact_features, rna_contact_features

    # 计算距离矩阵
    dist_matrix = cdist(protein_coords, rna_coords)

    # 为每个阈值计算接触特征
    for i, threshold in enumerate(thresholds):
        # 计算接触掩码
        contact_mask = (dist_matrix < threshold)

        # 计算每个蛋白质节点的接触比例
        protein_contact_features[:, i] = contact_mask.sum(axis=1) / len(rna_coords)

        # 计算每个RNA节点的接触比例
        rna_contact_features[:, i] = contact_mask.sum(axis=0) / len(protein_coords)

    return protein_contact_features, rna_contact_features


def assign_chemical_edge_weights(protein_features, rna_features, interface_pairs):
    """
    基于化学特性分配边权重

    参数:
    - protein_features: 蛋白质节点特征
    - rna_features: RNA节点特征
    - interface_pairs: 界面对列表

    返回:
    - edge_weights: 边权重数组
    - edge_features: 增强的边特征数组
    """
    # 氨基酸特性分类（基于物理化学特性）
    # 0:ALA, 1:ARG, 2:ASN, 3:ASP, 4:CYS, 5:GLN, 6:GLU, 7:GLY, 8:HIS, 9:ILE,
    # 10:LEU, 11:LYS, 12:MET, 13:PHE, 14:PRO, 15:SER, 16:THR, 17:TRP, 18:TYR, 19:VAL

    # 基于化学特性的相互作用强度矩阵（示例）
    # A, U, G, C, N(其他)的相互作用强度
    # 这是基于生物化学知识的示例矩阵，可根据实际情况调整
    interaction_matrix = np.ones((20, 5)) * 0.8  # 默认相互作用强度适中

    # 精氨酸(ARG)和鸟嘌呤(G)有强相互作用
    interaction_matrix[1, 2] = 1.5

    # 赖氨酸(LYS)和各种核苷酸
    interaction_matrix[11, :] = 1.2

    # 组氨酸(HIS)和腺嘌呤(A)
    interaction_matrix[8, 0] = 1.3

    # 天冬氨酸(ASP)和胞嘧啶(C)
    interaction_matrix[3, 3] = 1.2

    # 谷氨酸(GLU)和胞嘧啶(C)
    interaction_matrix[6, 3] = 1.2

    edge_weights = []
    edge_features = []

    for pair in interface_pairs:
        p_idx = pair['protein_idx']
        r_idx = pair['rna_idx']
        distance = pair['distance']

        # 提取残基类型
        p_type = np.argmax(protein_features[p_idx][:20]) if p_idx < len(protein_features) else 0
        r_type = np.argmax(rna_features[r_idx][:5]) if r_idx < len(rna_features) else 0

        # 基础权重随距离递减
        base_weight = 1.0 / (1.0 + distance / 5.0)  # 平滑衰减

        # 应用特定残基对的化学相互作用权重
        chemical_weight = interaction_matrix[p_type, r_type]

        # 最终权重
        weight = base_weight * chemical_weight

        # 边特征：距离编码 + 化学相互作用特征
        pos_encoding = positional_encoding(distance)

        # 添加额外特征：残基对类型
        # 为蛋白质残基建立one-hot向量
        protein_type = np.zeros(20)
        protein_type[p_type] = 1

        # 为RNA碱基建立one-hot向量
        rna_type = np.zeros(5)
        rna_type[r_type] = 1

        # 合并特征
        combined_feat = np.concatenate([
            pos_encoding,  # 距离编码 (16维)
            [weight],  # 相互作用权重 (1维)
            [distance],  # 原始距离 (1维)
            [chemical_weight],  # 化学权重 (1维)
            protein_type.astype(np.float32),  # 蛋白质残基类型 (20维)
            rna_type.astype(np.float32)  # RNA碱基类型 (5维)
        ])

        edge_weights.append(weight)
        edge_features.append(combined_feat)

    return np.array(edge_weights), np.array(edge_features)


def construct_interface_graph(protein_graph, rna_graph, interface_info, edge_offset=0):
    """
    构建界面图

    参数:
    - protein_graph: 蛋白质图
    - rna_graph: RNA图
    - interface_info: 界面信息
    - edge_offset: 边的索引偏移

    返回:
    - interface_graph: 界面图（NetworkX）
    """
    interface_graph = nx.Graph()

    # 获取节点特征和坐标
    protein_features = protein_graph.features
    protein_coords = protein_graph.coords
    rna_features = rna_graph.features
    rna_coords = rna_graph.coords

    # 获取界面对
    pairs = interface_info['pairs']

    # 分配化学边权重和特征
    edge_weights, edge_features = assign_chemical_edge_weights(
        protein_features, rna_features, pairs
    )

    # 添加边到界面图
    for i, pair in enumerate(pairs):
        p_idx = pair['protein_idx']
        r_idx = pair['rna_idx']
        r_idx_global = r_idx + len(protein_features)  # 全局索引

        # 添加边
        interface_graph.add_edge(
            p_idx,
            r_idx_global,
            weight=edge_weights[i],
            feature=edge_features[i],
            distance=pair['distance'],
            threshold=pair['threshold']
        )

    return interface_graph


def create_unified_multimodal_graph(wild_protein_graph, mutant_protein_graph, rna_graph,
                                    wild_interface, mutant_interface, mutation_pos):
    """
    创建统一的多模态图

    参数:
    - wild_protein_graph: 野生型蛋白质图
    - mutant_protein_graph: 突变型蛋白质图
    - rna_graph: RNA图
    - wild_interface: 野生型界面信息
    - mutant_interface: 突变型界面信息
    - mutation_pos: 突变位置

    返回:
    - unified_graph: 统一的多模态图
    """
    # 构建界面图
    wild_interface_graph = construct_interface_graph(
        wild_protein_graph, rna_graph, wild_interface
    )

    mutant_interface_graph = construct_interface_graph(
        mutant_protein_graph, rna_graph, mutant_interface
    )

    # 节点数量
    n_wild = len(wild_protein_graph.features)
    n_mutant = len(mutant_protein_graph.features)
    n_rna = len(rna_graph.features)

    # 特征最大维度
    max_feat_dim = max(
        wild_protein_graph.features.shape[1],
        mutant_protein_graph.features.shape[1],
        rna_graph.features.shape[1]
    )

    # 填充特征到统一维度
    def pad_features(features, target_dim):
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            return np.hstack([features, padding])
        return features

    wild_features = pad_features(wild_protein_graph.features, max_feat_dim)
    mutant_features = pad_features(mutant_protein_graph.features, max_feat_dim)
    rna_features = pad_features(rna_graph.features, max_feat_dim)

    # 合并节点特征
    all_features = np.vstack([
        wild_features,
        mutant_features,
        rna_features
    ])

    # 计算全局质心和相对坐标
    all_coords = np.vstack([
        wild_protein_graph.coords,
        mutant_protein_graph.coords,
        rna_graph.coords
    ])

    global_center = np.mean(all_coords, axis=0)

    wild_coords_rel = wild_protein_graph.coords - global_center
    mutant_coords_rel = mutant_protein_graph.coords - global_center
    rna_coords_rel = rna_graph.coords - global_center

    all_coords_rel = np.vstack([
        wild_coords_rel,
        mutant_coords_rel,
        rna_coords_rel
    ])

    # 创建掩码
    wild_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    wild_mask[:n_wild] = True

    mutant_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    mutant_mask[n_wild:n_wild + n_mutant] = True

    rna_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    rna_mask[n_wild + n_mutant:] = True

    # 界面掩码
    wild_interface_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    wild_interface_mask[:n_wild][wild_interface['protein_mask']] = True
    wild_interface_mask[n_wild + n_mutant:][wild_interface['rna_mask']] = True

    mutant_interface_mask = np.zeros(n_wild + n_mutant + n_rna, dtype=bool)
    mutant_interface_mask[n_wild:n_wild + n_mutant][mutant_interface['protein_mask']] = True
    mutant_interface_mask[n_wild + n_mutant:][mutant_interface['rna_mask']] = True

    interface_mask = wild_interface_mask | mutant_interface_mask

    # 根据全局坐标重建边

    # 1. 内部边偏移
    wild_edge_index = wild_protein_graph.edge_index.copy()
    mutant_edge_index = mutant_protein_graph.edge_index.copy() + n_wild
    rna_edge_index = rna_graph.edge_index.copy() + (n_wild + n_mutant)

    # 2. 界面边
    wild_interface_edge_index = []
    wild_interface_edge_attr = []

    for u, v, data in wild_interface_graph.edges(data=True):
        if v >= n_wild:  # RNA节点有全局偏移
            v = v - n_wild + (n_wild + n_mutant)
        wild_interface_edge_index.append([u, v])
        wild_interface_edge_index.append([v, u])  # 双向边

        edge_feat = data.get('feature', np.zeros(44))  # 界面边特征维度
        wild_interface_edge_attr.append(edge_feat)
        wild_interface_edge_attr.append(edge_feat)  # 双向边相同特征

    mutant_interface_edge_index = []
    mutant_interface_edge_attr = []

    for u, v, data in mutant_interface_graph.edges(data=True):
        u = u + n_wild  # 突变型蛋白质节点有全局偏移
        if v >= n_mutant:  # RNA节点有全局偏移
            v = v - n_mutant + (n_wild + n_mutant)
        mutant_interface_edge_index.append([u, v])
        mutant_interface_edge_index.append([v, u])  # 双向边

        edge_feat = data.get('feature', np.zeros(44))
        mutant_interface_edge_attr.append(edge_feat)
        mutant_interface_edge_attr.append(edge_feat)

    # 合并所有边索引和特征
    all_edge_indices = []
    all_edge_attrs = []
    all_edge_types = []

    # 野生型蛋白质内部边
    if len(wild_edge_index) > 0:
        all_edge_indices.append(wild_edge_index)
        all_edge_attrs.append(wild_protein_graph.edge_attr)
        all_edge_types.extend([0] * wild_edge_index.shape[1])

    # 突变型蛋白质内部边
    if len(mutant_edge_index) > 0:
        all_edge_indices.append(mutant_edge_index)
        all_edge_attrs.append(mutant_protein_graph.edge_attr)
        all_edge_types.extend([1] * mutant_edge_index.shape[1])

    # RNA内部边
    if len(rna_edge_index) > 0:
        all_edge_indices.append(rna_edge_index)
        all_edge_attrs.append(rna_graph.edge_attr)
        all_edge_types.extend([2] * rna_edge_index.shape[1])

    # 野生型界面边
    if wild_interface_edge_index:
        all_edge_indices.append(np.array(wild_interface_edge_index).T)
        all_edge_attrs.append(np.array(wild_interface_edge_attr))
        all_edge_types.extend([3] * len(wild_interface_edge_index))

    # 突变型界面边
    if mutant_interface_edge_index:
        all_edge_indices.append(np.array(mutant_interface_edge_index).T)
        all_edge_attrs.append(np.array(mutant_interface_edge_attr))
        all_edge_types.extend([4] * len(mutant_interface_edge_index))

    # 合并所有边
    edge_index = np.hstack(all_edge_indices) if all_edge_indices else np.zeros((2, 0), dtype=np.int64)

    # 由于边特征维度可能不同，需要统一
    # 找到最大的边特征维度
    max_edge_feat_dim = max([attr.shape[1] for attr in all_edge_attrs]) if all_edge_attrs else 0

    # 填充边特征
    padded_edge_attrs = []
    for attr in all_edge_attrs:
        if attr.shape[1] < max_edge_feat_dim:
            padding = np.zeros((attr.shape[0], max_edge_feat_dim - attr.shape[1]))
            padded_edge_attrs.append(np.hstack([attr, padding]))
        else:
            padded_edge_attrs.append(attr)

    edge_attr = np.vstack(padded_edge_attrs) if padded_edge_attrs else np.zeros((0, max_edge_feat_dim),
                                                                                dtype=np.float32)
    edge_type = np.array(all_edge_types, dtype=np.int32)

    # 突变位置
    # 在全局索引中找到突变位置
    if mutation_pos is not None:
        global_mutation_pos = mutation_pos
        mutant_mutation_pos = mutation_pos + n_wild
    else:
        global_mutation_pos = None
        mutant_mutation_pos = None

    # 构建统一图
    unified_graph = {
        'node_features': all_features,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'edge_type': edge_type,
        'pos': all_coords_rel,
        'global_center': global_center,
        'wild_mask': wild_mask,
        'mutant_mask': mutant_mask,
        'rna_mask': rna_mask,
        'interface_mask': interface_mask,
        'wild_interface_mask': wild_interface_mask,
        'mutant_interface_mask': mutant_interface_mask,
        'mutation_index': global_mutation_pos,
        'mutant_mutation_index': mutant_mutation_pos,
        'feature_dims': {
            'protein': wild_protein_graph.features.shape[1],
            'rna': rna_graph.features.shape[1],
            'unified': max_feat_dim,
            'edge': max_edge_feat_dim
        }
    }

    return unified_graph


def find_mutation_position(wild_features, mutant_features):
    """
    通过比较one-hot编码找到突变位点

    Args:
        wild_features: 野生型特征数组
        mutant_features: 突变型特征数组

    Returns:
        mutation_pos: 突变位点的索引
        mutation_info: 包含野生型和突变型氨基酸信息的字典
    """
    for i in range(len(wild_features)):
        # 只比较前20维的one-hot编码部分(氨基酸类型)
        if not np.array_equal(wild_features[i][:20], mutant_features[i][:20]):
            wild_aa = np.argmax(wild_features[i][:20])
            mutant_aa = np.argmax(mutant_features[i][:20])
            return i, {
                'wild_aa': wild_aa,
                'mutant_aa': mutant_aa,
                'position': i
            }
    return None, None

# ==================== 增强的数据集构建函数 ====================
def build_enhanced_multimodal_dataset(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                                      output_file, thresholds=[8.0],
                                      max_file_size_mb=1.0):
    """
    构建增强的多模态数据集，包含分段式接触特征

    参数:
    - csv_file: CSV文件路径
    - protein_dir: 蛋白质PDB目录
    - rna_dir: RNA PDB目录
    - pssm_file: PSSM文件路径
    - conservation_dir: 保守性得分目录
    - output_file: 输出文件路径
    - thresholds: 界面距离阈值列表
    - max_file_size_mb: 最大文件大小(MB)，超过此大小的文件将被跳过

    返回:
    - 数据集
    """
    logger.info("开始构建增强版多模态数据集（含分段式接触特征）")

    # 加载PSSM数据
    with open(pssm_file, 'rb') as f:
        pssm_dict = pickle.load(f)

    # 加载CSV数据
    df = pd.read_csv(csv_file)
    logger.info(f"加载了 {len(df)} 个突变数据")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 解析PDB文件
    parser = PDB.PDBParser(QUIET=True)

    # 构建数据集
    dataset = []
    skipped = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理突变"):
        pdb_id = row['PDB Id']
        chain_id = row['Mutated Chain']
        mutation = row['Mutation_PDB']
        ddg = row['DDGexp']
        mutation_pos = row.get('Mutation_Position', None)

        # 构建文件路径
        wild_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")
        mutant_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")
        rna_path = os.path.join(rna_dir, f"{pdb_id}_nucleic.pdb")

        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [wild_protein_path, mutant_protein_path, rna_path]):
            skipped.append((pdb_id, chain_id, mutation, "文件不存在"))
            continue

        # 检查文件大小
        file_sizes_mb = {
            'wild': os.path.getsize(wild_protein_path) / (1024 * 1024),
            'mutant': os.path.getsize(mutant_protein_path) / (1024 * 1024),
            'rna': os.path.getsize(rna_path) / (1024 * 1024)
        }

        if any(size > max_file_size_mb for size in file_sizes_mb.values()):
            too_large = [name for name, size in file_sizes_mb.items() if size > max_file_size_mb]
            logger.warning(
                f"跳过 {pdb_id}_{chain_id}_{mutation}: 文件 {','.join(too_large)} 超过大小限制 {max_file_size_mb}MB")
            skipped.append((pdb_id, chain_id, mutation, f"文件大小超过{max_file_size_mb}MB"))
            continue

        try:
            # 解析PDB结构
            wild_structure = parser.get_structure("wild", wild_protein_path)
            mutant_structure = parser.get_structure("mutant", mutant_protein_path)
            rna_structure = parser.get_structure("rna", rna_path)

            # 获取PSSM和保守性得分
            wild_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_wild_type.pssm", None)
            wild_conservation = load_conservation_scores(conservation_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")

            mutant_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_mutant_{mutation}.pssm", None)
            mutant_conservation = load_conservation_scores(conservation_dir,
                                                           f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")

            # 创建图结构
            wild_protein_graph = create_graph_from_structure(wild_structure, wild_pssm, wild_conservation)
            mutant_protein_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)
            rna_graph = create_rna_graph_from_structure(rna_structure, threshold=8.0)

            # 提取坐标
            wild_ca_coords = get_residue_ca_coords(wild_structure)
            mutant_ca_coords = get_residue_ca_coords(mutant_structure)
            rna_center_coords = get_rna_center_coords(rna_structure)

            # 转换为numpy数组
            wild_ca_coords_array = np.array(list(wild_ca_coords.values()))
            mutant_ca_coords_array = np.array(list(mutant_ca_coords.values()))
            rna_center_coords_array = np.array(list(rna_center_coords.values()))

            # 识别界面 (仍然使用原有的多尺度界面识别函数)
            wild_interface = identify_multiscale_interface(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_interface = identify_multiscale_interface(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )

            # 提取蛋白质基础特征
            wild_protein_features = np.array([data.get('feature', np.zeros(41))
                                              for _, data in wild_protein_graph.nodes(data=True)])
            mutant_protein_features = np.array([data.get('feature', np.zeros(41))
                                                for _, data in mutant_protein_graph.nodes(data=True)])

            # 找到突变位置和详细信息
            if mutation_pos is None:
                mutation_pos, mutation_details = find_mutation_position(wild_protein_features, mutant_protein_features)
            else:
                # 如果已经提供了突变位置，也获取详细信息
                if wild_protein_features[mutation_pos][:20].any() and mutant_protein_features[mutation_pos][:20].any():
                    wild_aa = np.argmax(wild_protein_features[mutation_pos][:20])
                    mutant_aa = np.argmax(mutant_protein_features[mutation_pos][:20])
                    mutation_details = {
                        'wild_aa': wild_aa,
                        'mutant_aa': mutant_aa,
                        'position': mutation_pos
                    }
                else:
                    mutation_details = None

            # 提取RNA的one-hot特征（前5维）
            rna_nodes = list(rna_graph.nodes())
            rna_onehot_features = np.zeros((len(rna_nodes), 5))

            for i, node in enumerate(rna_nodes):
                data = rna_graph.nodes[node]
                if 'feature' in data and len(data['feature']) >= 5:
                    # 只保留one-hot特征（通常是前5维）
                    rna_onehot_features[i, :] = data['feature'][:5]

            # 构建边和边特征
            wild_edges = []
            wild_edge_attr = []

            for u, v, data in wild_protein_graph.edges(data=True):
                # 获取节点索引
                u_idx = list(wild_protein_graph.nodes()).index(u)
                v_idx = list(wild_protein_graph.nodes()).index(v)

                wild_edges.append([u_idx, v_idx])
                wild_edges.append([v_idx, u_idx])  # 双向边

                edge_feat = data.get('feature', np.zeros(16))
                wild_edge_attr.append(edge_feat)
                wild_edge_attr.append(edge_feat)  # 双向边相同特征

            mutant_edges = []
            mutant_edge_attr = []

            for u, v, data in mutant_protein_graph.edges(data=True):
                u_idx = list(mutant_protein_graph.nodes()).index(u)
                v_idx = list(mutant_protein_graph.nodes()).index(v)

                mutant_edges.append([u_idx, v_idx])
                mutant_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                mutant_edge_attr.append(edge_feat)
                mutant_edge_attr.append(edge_feat)

            rna_edges = []
            rna_edge_attr = []

            for u, v, data in rna_graph.edges(data=True):
                u_idx = list(rna_graph.nodes()).index(u)
                v_idx = list(rna_graph.nodes()).index(v)

                rna_edges.append([u_idx, v_idx])
                rna_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                rna_edge_attr.append(edge_feat)
                rna_edge_attr.append(edge_feat)

            # 转换为numpy数组
            wild_edge_index = np.array(wild_edges).T if wild_edges else np.zeros((2, 0), dtype=np.int64)
            wild_edge_attr = np.array(wild_edge_attr) if wild_edge_attr else np.zeros((0, 16), dtype=np.float32)

            mutant_edge_index = np.array(mutant_edges).T if mutant_edges else np.zeros((2, 0), dtype=np.int64)
            mutant_edge_attr = np.array(mutant_edge_attr) if mutant_edge_attr else np.zeros((0, 16), dtype=np.float32)

            rna_edge_index = np.array(rna_edges).T if rna_edges else np.zeros((2, 0), dtype=np.int64)
            rna_edge_attr = np.array(rna_edge_attr) if rna_edge_attr else np.zeros((0, 16), dtype=np.float32)

            # =================================================================================
            # 新增: 创建分段式多尺度接触特征
            # =================================================================================
            # 将在data_loader中计算
            # wild_contact_features, rna_contact_features_wild, wild_contact_intensity, rna_contact_intensity_wild = create_segmented_contact_features(
            #     wild_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # mutant_contact_features, rna_contact_features_mutant, mutant_contact_intensity, rna_contact_intensity_mutant = create_segmented_contact_features(
            #     mutant_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # # 由于RNA需要同时与野生型和突变型计算接触特征，我们取两者的平均
            # rna_contact_features = (rna_contact_features_wild + rna_contact_features_mutant) / 2.0
            # rna_contact_intensity = (rna_contact_intensity_wild + rna_contact_intensity_mutant) / 2.0

            # 暂时不使用掉累积方式接触特征计算
            # wild_cumulative_features, rna_cumulative_features_wild = create_multiscale_contact_features(
            #     wild_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # mutant_cumulative_features, rna_cumulative_features_mutant = create_multiscale_contact_features(
            #     mutant_ca_coords_array, rna_center_coords_array, thresholds
            # )
            #
            # # 取平均
            # rna_cumulative_features = (rna_cumulative_features_wild + rna_cumulative_features_mutant) / 2.0

            # # 合并特征：基础特征 + 分段接触特征 + 接触强度
            # wild_protein_features = np.concatenate([
            #     wild_protein_features,  # 基础特征
            #     wild_contact_features,  # 分段接触特征
            #     wild_contact_intensity  # 接触强度
            # ], axis=1)
            #
            # mutant_protein_features = np.concatenate([
            #     mutant_protein_features,  # 基础特征
            #     mutant_contact_features,  # 分段接触特征
            #     mutant_contact_intensity  # 接触强度
            # ], axis=1)
            #
            # # RNA特征合并
            rna_features = rna_onehot_features
            # rna_features = np.concatenate([
            #     rna_onehot_features,  # 碱基one-hot编码(5维)
            #     rna_contact_features,  # 分段接触特征
            #     rna_contact_intensity  # 接触强度
            # ], axis=1)
            # =================================================================================

            # 记录特征维度信息（用于数据加载器中解析）
            feature_dims = {
                'base_dim': 41,  # 基础特征维度
                # 'segmented_contact_dim': len(thresholds) + 1,  # 分段接触特征维度
                # 'intensity_dim': 1  # 接触强度维度
                # 'cumulative_contact_dim': len(thresholds)    # 已移除累积接触特征
            }

            # 注意：如果要使用统一图，请将create_unified_graph设置为True
            create_unified_graph = False

            # 创建统一图（可选）
            unified_graph = create_unified_multimodal_graph(
                wild_protein_graph, mutant_protein_graph, rna_graph,
                wild_interface, mutant_interface, mutation_pos
            ) if create_unified_graph else None

            # 添加到数据集
            sample = {
                'unified_graph': unified_graph,
                'wild_graph': {
                    'features': wild_protein_features,
                    'coords': wild_ca_coords_array,
                    'edge_index': wild_edge_index,
                    'edge_attr': wild_edge_attr,
                    'interface': wild_interface
                },
                'mutant_graph': {
                    'features': mutant_protein_features,
                    'coords': mutant_ca_coords_array,
                    'edge_index': mutant_edge_index,
                    'edge_attr': mutant_edge_attr,
                    'interface': mutant_interface
                },
                'rna_graph': {
                    'features': rna_features,
                    'coords': rna_center_coords_array,
                    'edge_index': rna_edge_index,
                    'edge_attr': rna_edge_attr
                },
                'metadata': {
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'mutation': mutation,
                    'ddg': ddg,
                    'mutation_pos': mutation_pos,
                    'mutation_details': mutation_details,
                    'feature_dims': feature_dims,
                    'thresholds': thresholds,
                    # 'enhanced': True  # 标记使用了增强特征
                }
            }

            dataset.append(sample)

            # 每50个样本打印一次进度
            if len(dataset) % 50 == 0:
                logger.info(f"已处理 {len(dataset)} 个样本，跳过 {len(skipped)} 个样本")

                # 简单的界面统计
                avg_wild_contacts = np.mean([len(s['wild_graph']['interface']['pairs']) for s in dataset])
                avg_mutant_contacts = np.mean([len(s['mutant_graph']['interface']['pairs']) for s in dataset])

                logger.info(f"平均野生型界面接触数: {avg_wild_contacts:.2f}")
                logger.info(f"平均突变型界面接触数: {avg_mutant_contacts:.2f}")

        except Exception as e:
            logger.error(f"处理样本 {pdb_id}_{chain_id}_{mutation} 出错: {str(e)}")
            skipped.append((pdb_id, chain_id, mutation, str(e)))
            continue

    # 保存数据集
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    # 保存跳过的样本
    with open(os.path.join(os.path.dirname(output_file), "skipped_samples.txt"), 'w') as f:
        for item in skipped:
            f.write(f"{item[0]}_{item[1]}_{item[2]}: {item[3]}\n")


    logger.info(f"数据集构建完成，共 {len(dataset)} 个样本，保存至 {output_file}")
    logger.info(f"跳过 {len(skipped)} 个样本，详情见 skipped_samples.txt")

    return dataset






def build_multimodal_dataset_simplified(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                                        output_file, thresholds=[8.0, 12.0, 16.0], max_file_size_mb=1.0):
    """
    构建简化版多模态数据集，RNA特征仅包含one-hot编码和接触特征

    参数:
    - csv_file: CSV文件路径
    - protein_dir: 蛋白质PDB目录
    - rna_dir: RNA PDB目录
    - pssm_file: PSSM文件路径
    - conservation_dir: 保守性得分目录
    - output_file: 输出文件路径
    - thresholds: 界面距离阈值列表
    - max_file_size_mb: 最大文件大小(MB)，超过此大小的文件将被跳过

    返回:
    - 数据集
    """
    logger.info("开始构建简化版多模态数据集（RNA仅包含one-hot和接触特征）")

    # 加载PSSM数据
    with open(pssm_file, 'rb') as f:
        pssm_dict = pickle.load(f)

    # 加载CSV数据
    df = pd.read_csv(csv_file)
    logger.info(f"加载了 {len(df)} 个突变数据")

    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 解析PDB文件
    parser = PDB.PDBParser(QUIET=True)

    # 构建数据集
    dataset = []
    skipped = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理突变"):
        pdb_id = row['PDB Id']
        chain_id = row['Mutated Chain']
        mutation = row['Mutation_PDB']
        ddg = row['DDGexp']

        # 构建文件路径
        wild_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")
        mutant_protein_path = os.path.join(protein_dir, f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")
        rna_path = os.path.join(rna_dir, f"{pdb_id}_nucleic.pdb")

        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [wild_protein_path, mutant_protein_path, rna_path]):
            skipped.append((pdb_id, chain_id, mutation, "文件不存在"))
            continue

        # 检查文件大小
        file_sizes_mb = {
            'wild': os.path.getsize(wild_protein_path) / (1024 * 1024),
            'mutant': os.path.getsize(mutant_protein_path) / (1024 * 1024),
            'rna': os.path.getsize(rna_path) / (1024 * 1024)
        }

        if any(size > max_file_size_mb for size in file_sizes_mb.values()):
            too_large = [name for name, size in file_sizes_mb.items() if size > max_file_size_mb]
            logger.warning(
                f"跳过 {pdb_id}_{chain_id}_{mutation}: 文件 {','.join(too_large)} 超过大小限制 {max_file_size_mb}MB")
            skipped.append((pdb_id, chain_id, mutation, f"文件大小超过{max_file_size_mb}MB"))
            continue

        try:
            # 解析PDB结构
            wild_structure = parser.get_structure("wild", wild_protein_path)
            mutant_structure = parser.get_structure("mutant", mutant_protein_path)
            rna_structure = parser.get_structure("rna", rna_path)

            # 获取PSSM和保守性得分
            wild_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_wild_type.pssm", None)
            wild_conservation = load_conservation_scores(conservation_dir, f"{pdb_id}_{chain_id}_wild_type.pdb")

            mutant_pssm = pssm_dict.get(f"{pdb_id}_{chain_id}_mutant_{mutation}.pssm", None)
            mutant_conservation = load_conservation_scores(conservation_dir,
                                                           f"{pdb_id}_{chain_id}_mutant_{mutation}.pdb")

            # 创建图结构
            wild_protein_graph = create_graph_from_structure(wild_structure, wild_pssm, wild_conservation)
            mutant_protein_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)
            rna_graph = create_rna_graph_from_structure(rna_structure, threshold=8.0)

            # 提取坐标
            wild_ca_coords = get_residue_ca_coords(wild_structure)
            mutant_ca_coords = get_residue_ca_coords(mutant_structure)
            rna_center_coords = get_rna_center_coords(rna_structure)

            # 转换为numpy数组
            wild_ca_coords_array = np.array(list(wild_ca_coords.values()))
            mutant_ca_coords_array = np.array(list(mutant_ca_coords.values()))
            rna_center_coords_array = np.array(list(rna_center_coords.values()))

            # 识别界面
            wild_interface = identify_multiscale_interface(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_interface = identify_multiscale_interface(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )

            # 找到突变位置
            mutation_pos = None
            for i, (node1, node2) in enumerate(zip(wild_protein_graph.nodes(), mutant_protein_graph.nodes())):
                data1 = wild_protein_graph.nodes[node1]
                data2 = mutant_protein_graph.nodes[node2]

                if 'feature' in data1 and 'feature' in data2:
                    feat1 = data1['feature']
                    feat2 = data2['feature']

                    if len(feat1) >= 20 and len(feat2) >= 20 and not np.array_equal(feat1[:20], feat2[:20]):
                        mutation_pos = i
                        break

            # 获取图特征和拓扑结构
            wild_protein_features = np.array([data.get('feature', np.zeros(41))
                                              for _, data in wild_protein_graph.nodes(data=True)])
            mutant_protein_features = np.array([data.get('feature', np.zeros(41))
                                                for _, data in mutant_protein_graph.nodes(data=True)])

            # ==== 关键修改：提取RNA的one-hot特征（前5维） ====
            rna_nodes = list(rna_graph.nodes())
            rna_onehot_features = np.zeros((len(rna_nodes), 5))

            for i, node in enumerate(rna_nodes):
                data = rna_graph.nodes[node]
                if 'feature' in data and len(data['feature']) >= 5:
                    # 只保留one-hot特征（通常是前5维）
                    rna_onehot_features[i, :] = data['feature'][:5]

            # 构建边和边特征
            wild_edges = []
            wild_edge_attr = []

            for u, v, data in wild_protein_graph.edges(data=True):
                # 获取节点索引
                u_idx = list(wild_protein_graph.nodes()).index(u)
                v_idx = list(wild_protein_graph.nodes()).index(v)

                wild_edges.append([u_idx, v_idx])
                wild_edges.append([v_idx, u_idx])  # 双向边

                edge_feat = data.get('feature', np.zeros(16))
                wild_edge_attr.append(edge_feat)
                wild_edge_attr.append(edge_feat)  # 双向边相同特征

            mutant_edges = []
            mutant_edge_attr = []

            for u, v, data in mutant_protein_graph.edges(data=True):
                u_idx = list(mutant_protein_graph.nodes()).index(u)
                v_idx = list(mutant_protein_graph.nodes()).index(v)

                mutant_edges.append([u_idx, v_idx])
                mutant_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                mutant_edge_attr.append(edge_feat)
                mutant_edge_attr.append(edge_feat)

            rna_edges = []
            rna_edge_attr = []

            for u, v, data in rna_graph.edges(data=True):
                u_idx = list(rna_graph.nodes()).index(u)
                v_idx = list(rna_graph.nodes()).index(v)

                rna_edges.append([u_idx, v_idx])
                rna_edges.append([v_idx, u_idx])

                edge_feat = data.get('feature', np.zeros(16))
                rna_edge_attr.append(edge_feat)
                rna_edge_attr.append(edge_feat)

            # 转换为numpy数组
            wild_edge_index = np.array(wild_edges).T if wild_edges else np.zeros((2, 0), dtype=np.int64)
            wild_edge_attr = np.array(wild_edge_attr) if wild_edge_attr else np.zeros((0, 16), dtype=np.float32)

            mutant_edge_index = np.array(mutant_edges).T if mutant_edges else np.zeros((2, 0), dtype=np.int64)
            mutant_edge_attr = np.array(mutant_edge_attr) if mutant_edge_attr else np.zeros((0, 16), dtype=np.float32)

            rna_edge_index = np.array(rna_edges).T if rna_edges else np.zeros((2, 0), dtype=np.int64)
            rna_edge_attr = np.array(rna_edge_attr) if rna_edge_attr else np.zeros((0, 16), dtype=np.float32)

            # 创建多尺度接触特征
            wild_contact_features, rna_contact_features_wild = create_multiscale_contact_features(
                wild_ca_coords_array, rna_center_coords_array, thresholds
            )

            mutant_contact_features, rna_contact_features_mutant = create_multiscale_contact_features(
                mutant_ca_coords_array, rna_center_coords_array, thresholds
            )

            # 由于RNA需要同时与野生型和突变型计算接触特征，我们取两者的平均
            rna_contact_features = (rna_contact_features_wild + rna_contact_features_mutant) / 2.0

            # 合并特征
            wild_protein_features = np.concatenate([wild_protein_features, wild_contact_features], axis=1)
            mutant_protein_features = np.concatenate([mutant_protein_features, mutant_contact_features], axis=1)

            # ==== 关键修改：RNA仅使用one-hot特征和接触特征 ====
            rna_features = np.concatenate([rna_onehot_features, rna_contact_features], axis=1)

            # 创建统一图（注释掉，但保持兼容性）
            unified_graph = None

            # 添加到数据集
            sample = {
                'unified_graph': unified_graph,  # 目前为None，保持兼容性
                'wild_graph': {
                    'features': wild_protein_features,
                    'coords': wild_ca_coords_array,
                    'edge_index': wild_edge_index,
                    'edge_attr': wild_edge_attr,
                    'interface': wild_interface
                },
                'mutant_graph': {
                    'features': mutant_protein_features,
                    'coords': mutant_ca_coords_array,
                    'edge_index': mutant_edge_index,
                    'edge_attr': mutant_edge_attr,
                    'interface': mutant_interface
                },
                'rna_graph': {
                    'features': rna_features,  # 简化的RNA特征
                    'coords': rna_center_coords_array,
                    'edge_index': rna_edge_index,
                    'edge_attr': rna_edge_attr
                },
                'metadata': {
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'mutation': mutation,
                    'ddg': ddg,
                    'mutation_pos': mutation_pos,
                    'contact_feature_dim': len(thresholds),  # 记录接触特征维度
                    'rna_simplified': True  # 标记使用了简化RNA特征
                }
            }

            dataset.append(sample)

            # 每50个样本打印一次进度
            if len(dataset) % 50 == 0:
                logger.info(f"已处理 {len(dataset)} 个样本，跳过 {len(skipped)} 个样本")

                # 简单的界面统计
                avg_wild_contacts = np.mean([len(s['wild_graph']['interface']['pairs']) for s in dataset])
                avg_mutant_contacts = np.mean([len(s['mutant_graph']['interface']['pairs']) for s in dataset])

                logger.info(f"平均野生型界面接触数: {avg_wild_contacts:.2f}")
                logger.info(f"平均突变型界面接触数: {avg_mutant_contacts:.2f}")

        except Exception as e:
            logger.error(f"处理样本 {pdb_id}_{chain_id}_{mutation} 出错: {str(e)}")
            skipped.append((pdb_id, chain_id, mutation, str(e)))
            continue

    # 保存数据集
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    # 保存跳过的样本
    with open(os.path.join(os.path.dirname(output_file), "skipped_samples.txt"), 'w') as f:
        for item in skipped:
            f.write(f"{item[0]}_{item[1]}_{item[2]}: {item[3]}\n")

    logger.info(f"数据集构建完成，共 {len(dataset)} 个样本，保存至 {output_file}")
    logger.info(f"跳过 {len(skipped)} 个样本，详情见 skipped_samples.txt")
    logger.info(f"RNA特征维度: {rna_features.shape[1]} (one-hot特征:5, 接触特征:{len(thresholds)})")

    return dataset

# 原始数据集构建函数，保持兼容性
def build_multimodal_dataset(csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
                             output_file, thresholds=[8.0, 12.0, 16.0], max_file_size_mb=1.0):
    """
    构建增强的多模态数据集

    参数:
    - csv_file: CSV文件路径
    - protein_dir: 蛋白质PDB目录
    - rna_dir: RNA PDB目录
    - pssm_file: PSSM文件路径
    - conservation_dir: 保守性得分目录
    - output_file: 输出文件路径
    - thresholds: 界面距离阈值列表
    - max_file_size_mb: 最大文件大小(MB)，超过此大小的文件将被跳过

    返回:
    - 数据集
    """
    # 调用增强版本，保持向后兼容性
    return build_multimodal_dataset_simplified(
        csv_file, protein_dir, rna_dir, pssm_file, conservation_dir,
        output_file, thresholds, max_file_size_mb
    )


# ==================== 可视化函数 ====================

def visualize_interface_contacts(sample, output_dir="interface_viz"):
    """
    可视化蛋白质-RNA界面接触

    参数:
    - sample: 数据集样本
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    pdb_id = sample['metadata']['pdb_id']
    chain_id = sample['metadata']['chain_id']
    mutation = sample['metadata']['mutation']
    ddg = sample['metadata']['ddg']
    mutation_pos = sample['metadata']['mutation_pos']

    # 提取界面信息
    wild_interface = sample['wild_graph']['interface']
    mutant_interface = sample['mutant_graph']['interface']

    wild_coords = sample['wild_graph']['coords']
    mutant_coords = sample['mutant_graph']['coords']
    rna_coords = sample['rna_graph']['coords']

    # 绘制接触图
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # 野生型接触图
    wild_contact = np.zeros((len(wild_coords), len(rna_coords)))
    for pair in wild_interface['pairs']:
        wild_contact[pair['protein_idx'], pair['rna_idx']] = 1

    axes[0].imshow(wild_contact, cmap='viridis')
    axes[0].set_title(f'Wild Type Interface Contacts\n{pdb_id}_{chain_id} - {len(wild_interface["pairs"])} contacts')
    axes[0].set_xlabel('RNA Residues')
    axes[0].set_ylabel('Protein Residues')

    # 标记突变位置
    if mutation_pos is not None:
        axes[0].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)

    # 突变型接触图
    mutant_contact = np.zeros((len(mutant_coords), len(rna_coords)))
    for pair in mutant_interface['pairs']:
        mutant_contact[pair['protein_idx'], pair['rna_idx']] = 1

    axes[1].imshow(mutant_contact, cmap='viridis')
    axes[1].set_title(
        f'Mutant Type Interface Contacts\n{mutation} (DDG: {ddg:.2f}) - {len(mutant_interface["pairs"])} contacts')
    axes[1].set_xlabel('RNA Residues')
    axes[1].set_ylabel('Protein Residues')

    # 标记突变位置
    if mutation_pos is not None:
        axes[1].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_interface.png"), dpi=300)
    plt.close()

    # 绘制差异图
    diff_contact = mutant_contact - wild_contact

    plt.figure(figsize=(10, 8))
    plt.imshow(diff_contact, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Mutant - Wild')
    plt.title(f'Interface Contact Differences\n{pdb_id}_{chain_id} {mutation} (DDG: {ddg:.2f})')
    plt.xlabel('RNA Residues')
    plt.ylabel('Protein Residues')

    # 标记突变位置
    if mutation_pos is not None:
        plt.axhline(y=mutation_pos, color='black', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_interface_diff.png"), dpi=300)
    plt.close()

    # 可视化接触特征（新增）
    contact_dim = sample['metadata'].get('contact_feature_dim', 0)
    if contact_dim > 0:
        # 提取接触特征
        wild_features = sample['wild_graph']['features']
        mutant_features = sample['mutant_graph']['features']

        wild_contact_features = wild_features[:, -contact_dim:]
        mutant_contact_features = mutant_features[:, -contact_dim:]

        # 绘制接触特征热图
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # 野生型接触特征
        im0 = axes[0].imshow(wild_contact_features, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Wild Type Contact Features\n{pdb_id}_{chain_id}')
        axes[0].set_xlabel('Threshold Levels')
        axes[0].set_ylabel('Protein Residues')
        fig.colorbar(im0, ax=axes[0])

        # 突变型接触特征
        im1 = axes[1].imshow(mutant_contact_features, cmap='viridis', aspect='auto')
        axes[1].set_title(f'Mutant Type Contact Features\n{mutation} (DDG: {ddg:.2f})')
        axes[1].set_xlabel('Threshold Levels')
        axes[1].set_ylabel('Protein Residues')
        fig.colorbar(im1, ax=axes[1])

        # 标记突变位置
        if mutation_pos is not None:
            axes[0].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=mutation_pos, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_contact_features.png"), dpi=300)
        plt.close()

        # 绘制接触特征差异
        diff_features = mutant_contact_features - wild_contact_features
        plt.figure(figsize=(10, 8))
        im = plt.imshow(diff_features, cmap='coolwarm', aspect='auto')
        plt.colorbar(im, label='Mutant - Wild')
        plt.title(f'Contact Feature Differences\n{pdb_id}_{chain_id} {mutation} (DDG: {ddg:.2f})')
        plt.xlabel('Threshold Levels')
        plt.ylabel('Protein Residues')

        # 标记突变位置
        if mutation_pos is not None:
            plt.axhline(y=mutation_pos, color='black', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{pdb_id}_{chain_id}_{mutation}_contact_features_diff.png"), dpi=300)
        plt.close()


def analyze_dataset(dataset, output_dir="dataset_analysis"):
    """
    分析数据集统计信息

    参数:
    - dataset: 数据集
    - output_dir: 输出目录

    返回:
    - stats: 统计信息字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 基本统计
    n_samples = len(dataset)
    pdb_ids = set([s['metadata']['pdb_id'] for s in dataset])
    chains = set([s['metadata']['chain_id'] for s in dataset])

    # 节点数统计
    protein_nodes = [len(s['wild_graph']['features']) for s in dataset]
    rna_nodes = [len(s['rna_graph']['features']) for s in dataset]

    # 特征维度
    protein_feat_dims = set([s['wild_graph']['features'].shape[1] for s in dataset])
    rna_feat_dims = set([s['rna_graph']['features'].shape[1] for s in dataset])

    # 检查是否有unified_graph
    has_unified = all(s['unified_graph'] is not None for s in dataset)
    unified_feat_dims = set() if not has_unified else set(
        [s['unified_graph']['node_features'].shape[1] for s in dataset])

    # 边数统计
    edge_counts = [] if not has_unified else [s['unified_graph']['edge_index'].shape[1] for s in dataset]

    # 边类型分布
    edge_types = []
    if has_unified:
        for s in dataset:
            edge_types.extend(s['unified_graph']['edge_type'].tolist())

    from collections import defaultdict
    type_counts = defaultdict(int)
    for t in edge_types:
        type_counts[t] += 1

    # DDG分布
    ddg_values = [s['metadata']['ddg'] for s in dataset]

    # 界面接触数量
    wild_interface_counts = [len(s['wild_graph']['interface']['pairs']) for s in dataset]
    mutant_interface_counts = [len(s['mutant_graph']['interface']['pairs']) for s in dataset]

    # 检查是否有接触特征
    has_contact_features = 'contact_feature_dim' in dataset[0]['metadata']
    contact_dim = dataset[0]['metadata'].get('contact_feature_dim', 0) if has_contact_features else 0

    # 可视化DDG分布
    plt.figure(figsize=(10, 6))
    plt.hist(ddg_values, bins=20, alpha=0.7)
    plt.axvline(x=np.mean(ddg_values), color='r', linestyle='--', alpha=0.7, label=f'平均值: {np.mean(ddg_values):.2f}')
    plt.title('DDG分布')
    plt.xlabel('DDG值')
    plt.ylabel('频率')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "ddg_distribution.png"), dpi=300)
    plt.close()

    # 可视化界面接触数量分布
    plt.figure(figsize=(10, 6))
    plt.hist(wild_interface_counts, bins=20, alpha=0.5, label='Wild Type')
    plt.hist(mutant_interface_counts, bins=20, alpha=0.5, label='Mutant')
    plt.title('界面接触数量分布')
    plt.xlabel('接触数量')
    plt.ylabel('频率')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "interface_contacts_distribution.png"), dpi=300)
    plt.close()

    # 如果有接触特征，可视化接触特征分布
    if has_contact_features and contact_dim > 0:
        # 提取所有样本的接触特征
        all_wild_contacts = np.vstack([s['wild_graph']['features'][:, -contact_dim:].mean(axis=0) for s in dataset])
        all_mutant_contacts = np.vstack([s['mutant_graph']['features'][:, -contact_dim:].mean(axis=0) for s in dataset])

        # 绘制平均接触特征
        plt.figure(figsize=(10, 6))
        thresholds = [f'T{i + 1}' for i in range(contact_dim)]
        plt.bar(np.arange(contact_dim) - 0.2, all_wild_contacts.mean(axis=0), width=0.4, alpha=0.7, label='Wild Type')
        plt.bar(np.arange(contact_dim) + 0.2, all_mutant_contacts.mean(axis=0), width=0.4, alpha=0.7, label='Mutant')
        plt.xticks(np.arange(contact_dim), thresholds)
        plt.title('平均接触特征分布')
        plt.xlabel('阈值等级')
        plt.ylabel('平均接触比例')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "contact_features_distribution.png"), dpi=300)
        plt.close()

    # 打印统计信息
    logger.info(f"统计信息摘要:")
    logger.info(f"样本数量: {n_samples}")
    logger.info(f"独特PDB数量: {len(pdb_ids)}")
    logger.info(f"独特链数量: {len(chains)}")
    logger.info(f"平均蛋白质节点数: {np.mean(protein_nodes):.2f}")
    logger.info(f"平均RNA节点数: {np.mean(rna_nodes):.2f}")
    logger.info(f"蛋白质特征维度: {protein_feat_dims}")
    logger.info(f"RNA特征维度: {rna_feat_dims}")
    if has_unified:
        logger.info(f"统一图特征维度: {unified_feat_dims}")
        logger.info(f"平均边数: {np.mean(edge_counts):.2f}")
        logger.info(f"边类型分布: {type_counts}")
    logger.info(f"DDG值范围: [{min(ddg_values):.2f}, {max(ddg_values):.2f}]")
    logger.info(f"DDG均值: {np.mean(ddg_values):.2f}")
    logger.info(f"DDG标准差: {np.std(ddg_values):.2f}")
    logger.info(f"平均界面对数量: {np.mean(wild_interface_counts):.2f}")
    if has_contact_features:
        logger.info(f"接触特征维度: {contact_dim}")

    # 可视化几个样本的界面
    visualize_samples = [dataset[i] for i in np.linspace(0, len(dataset) - 1, 5, dtype=int)]
    for sample in visualize_samples:
        visualize_interface_contacts(sample, output_dir)

    # 保存统计数据
    stats = {
        'n_samples': n_samples,
        'pdb_ids': pdb_ids,
        'chains': chains,
        'protein_nodes': protein_nodes,
        'rna_nodes': rna_nodes,
        'protein_feat_dims': protein_feat_dims,
        'rna_feat_dims': rna_feat_dims,
        'unified_feat_dims': unified_feat_dims if has_unified else None,
        'edge_counts': edge_counts if has_unified else None,
        'type_counts': type_counts if has_unified else None,
        'ddg_values': ddg_values,
        'wild_interface_counts': wild_interface_counts,
        'mutant_interface_counts': mutant_interface_counts,
        'has_contact_features': has_contact_features,
        'contact_feature_dim': contact_dim
    }

    with open(os.path.join(output_dir, "dataset_statistics.pkl"), 'wb') as f:
        pickle.dump(stats, f)

    logger.info(f"可视化结果已保存至: {output_dir}")
    logger.info(f"统计结果已保存至: {os.path.join(output_dir, 'dataset_statistics.pkl')}")

    return stats


# ==================== 样本分析函数 ====================

def analyze_sample(dataset, sample_idx=0):
    """
    详细分析单个样本

    参数:
    - dataset: 数据集
    - sample_idx: 样本索引
    """
    sample = dataset[sample_idx]

    logger.info("\n==================================================")
    logger.info(f"样本 {sample_idx} 详细分析")
    logger.info("==================================================")

    # 元数据
    metadata = sample['metadata']
    logger.info(f"PDB ID: {metadata['pdb_id']}")
    logger.info(f"链: {metadata['chain_id']}")
    logger.info(f"突变: {metadata['mutation']}")
    logger.info(f"DDG: {metadata['ddg']}")
    logger.info(f"突变位置: {metadata['mutation_pos']}")

    # 野生型图信息
    logger.info(f"\n野生型图:")
    logger.info(f"蛋白质节点数: {len(sample['wild_graph']['features'])}")
    logger.info(f"蛋白质特征维度: {sample['wild_graph']['features'].shape[1]}")
    logger.info(f"RNA节点数: {len(sample['rna_graph']['features'])}")
    logger.info(f"RNA特征维度: {sample['rna_graph']['features'].shape[1]}")
    logger.info(f"界面对数: {len(sample['wild_graph']['interface']['pairs'])}")

    # 突变型图信息
    logger.info(f"\n突变型图:")
    logger.info(f"蛋白质节点数: {len(sample['mutant_graph']['features'])}")
    logger.info(f"界面对数: {len(sample['mutant_graph']['interface']['pairs'])}")

    # 接触特征信息
    if 'contact_feature_dim' in metadata:
        contact_dim = metadata['contact_feature_dim']
        logger.info(f"\n接触特征:")
        logger.info(f"接触特征维度: {contact_dim}")

        # 提取接触特征
        wild_contact_features = sample['wild_graph']['features'][:, -contact_dim:]
        mutant_contact_features = sample['mutant_graph']['features'][:, -contact_dim:]

        logger.info(f"野生型平均接触特征: {wild_contact_features.mean(axis=0)}")
        logger.info(f"突变型平均接触特征: {mutant_contact_features.mean(axis=0)}")

        # 计算接触特征差异
        logger.info(f"接触特征平均差异: {(mutant_contact_features - wild_contact_features).mean(axis=0)}")

    # 统一图信息(如果存在)
    if sample['unified_graph'] is not None:
        unified = sample['unified_graph']
        logger.info(f"\n统一图:")
        logger.info(f"总节点数: {unified['node_features'].shape[0]}")
        logger.info(f"特征维度: {unified['node_features'].shape[1]}")
        logger.info(f"总边数: {unified['edge_index'].shape[1]}")

        # 边类型分布
        edge_type_counts = {}
        for t in range(5):  # 0: 野生型内部, 1: 突变型内部, 2: RNA内部, 3: 野生型界面, 4: 突变型界面
            edge_type_counts[t] = np.sum(unified['edge_type'] == t)

        logger.info(f"\n边类型分布:")
        logger.info(f"野生型蛋白质内部: {edge_type_counts.get(0, 0)}")
        logger.info(f"突变型蛋白质内部: {edge_type_counts.get(1, 0)}")
        logger.info(f"RNA内部: {edge_type_counts.get(2, 0)}")
        logger.info(f"野生型界面: {edge_type_counts.get(3, 0)}")
        logger.info(f"突变型界面: {edge_type_counts.get(4, 0)}")

        # 特征维度信息
        logger.info(f"\n特征维度信息:")
        logger.info(f"原始蛋白质特征维度: {unified['feature_dims']['protein']}")
        logger.info(f"原始RNA特征维度: {unified['feature_dims']['rna']}")
        logger.info(f"统一后特征维度: {unified['feature_dims']['unified']}")

    return sample


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 目录设置
    COMPLEX_DIR = "./Dataset/S394"  # 复合物PDB文件目录
    PROTEIN_DIR = "./processed_pdbs/protein_chains"  # 分离的蛋白质PDB
    RNA_DIR = "./processed_pdbs/nucleic_chains"  # 分离的RNA PDB
    PSSM_FILE = "./Dataset/PSSM_394/pssm_s394.pkl"  # PSSM文件
    CONSERVATION_DIR = "./Dataset/cons_s394"  # 保守性得分目录
    CSV_FILE = "./Dataset/S394.csv"  # 数据集CSV文件

    # 输出设置
    OUTPUT_DIR = "./dataset"

    # 构建增强版数据集
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "enhanced_protein_rna_dataset_with_contacts.pkl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # dataset1 = build_multimodal_dataset_enhanced(
    #     csv_file=CSV_FILE,
    #     protein_dir=PROTEIN_DIR,
    #     rna_dir=RNA_DIR,
    #     pssm_file=PSSM_FILE,
    #     conservation_dir=CONSERVATION_DIR,
    #     output_file=OUTPUT_FILE,
    #     thresholds=[8.0, 12.0, 16.0],
    #     max_file_size_mb=1.0
    # )

    # # 构建简化版数据集
    # OUTPUT_FILE_SIMPLIFIED = os.path.join(OUTPUT_DIR, "simplified_protein_rna_dataset.pkl")
    # dataset = build_enhanced_multimodal_dataset(
    #     csv_file=CSV_FILE,
    #     protein_dir=PROTEIN_DIR,
    #     rna_dir=RNA_DIR,
    #     pssm_file=PSSM_FILE,
    #     conservation_dir=CONSERVATION_DIR,
    #     output_file=OUTPUT_FILE_SIMPLIFIED,
    #     thresholds=[7.0, 8.0],
    #     max_file_size_mb=1.0
    # )

    # 建议更新
    OUTPUT_FILE_ENHANCED = os.path.join(OUTPUT_DIR, "protein_rna_dataset.pkl")
    dataset = build_enhanced_multimodal_dataset(
        csv_file=CSV_FILE,
        protein_dir=PROTEIN_DIR,
        rna_dir=RNA_DIR,
        pssm_file=PSSM_FILE,
        conservation_dir=CONSERVATION_DIR,
        output_file=OUTPUT_FILE_ENHANCED,  # 更新文件名
        # thresholds=DEFAULT_CONTACT_THRESHOLDS,  # deprecated
        max_file_size_mb=5.0
    )

    # 分析数据集
    # analyze_dataset(dataset1, output_dir="dataset_analysis_enhanced")
    analyze_dataset(dataset, output_dir="dataset_analysis_enhanced")

    # 详细分析几个样本
    sample_indices = [0, len(dataset) // 4, len(dataset) // 2, 3 * len(dataset) // 4, len(dataset) - 1]
    for idx in sample_indices:
        analyze_sample(dataset, idx)


if __name__ == "__main__":
    main()