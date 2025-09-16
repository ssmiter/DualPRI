import logging
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from Bio import PDB
import pickle
import re
import pandas as pd
from tqdm import tqdm

from config import ENCODING_DIM

def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords



aa_dict = {
    'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLU': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}


def get_residue_ca_coords(structure):
    ca_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords[(chain.id, residue.id)] = residue['CA'].coord
    return ca_coords


def positional_encoding(distance, dimension=16):
    encoding = np.zeros(dimension)
    for i in range(dimension):
        encoding[i] = np.sin(distance / (10000 ** ((2 * i) / dimension))) if i % 2 == 0 else np.cos(
            distance / (10000 ** ((2 * i) / dimension)))
    return encoding


def load_conservation_scores(directory, filename):
    conservation_file = os.path.join(directory, filename.replace('.pdb', '_conservation.npy'))
    if os.path.exists(conservation_file):
        return np.load(conservation_file)
    return None


def create_graph_from_structure(structure, pssm_scores, conservation_scores, threshold=8.0, encoding_dim=ENCODING_DIM):
    ca_coords = get_residue_ca_coords(structure)
    graph = nx.Graph()

    # Add nodes with combined features
    for i, ((chain_id, residue_id), coord) in enumerate(ca_coords.items()):
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()
        one_hot = np.array(aa_dict.get(resname, [0] * 20))  # Convert one-hot encoding to a NumPy array
        pssm = np.array(pssm_scores[i]) if pssm_scores is not None and i < len(pssm_scores) else np.zeros(20)
        # 归一化PSSM得分
        pssm = np.clip(pssm, -10, 10)
        pssm = (pssm + 10) / 20
        conservation = np.array([conservation_scores[i]]) if conservation_scores is not None and i < len(
            conservation_scores) else np.zeros(1)
        features = np.concatenate((one_hot, pssm, conservation))  # Concatenate features
        graph.add_node((chain_id, residue_id), feature=features)

    # Add edges based on distance threshold
    for (chain_id1, res_id1), coord1 in ca_coords.items():
        for (chain_id2, res_id2), coord2 in ca_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1), (chain_id2, res_id2), feature=edge_feature)

    return graph


def process_pdb_files(directory, pssm_pickle_file, conservation_directory, progress_bar=True):
    parser = PDB.PDBParser(QUIET=True)
    graphs = []

    wild_type_files = {}
    mutant_files = {}

    # Load PSSM scores
    with open(pssm_pickle_file, 'rb') as file:
        pssm_scores_dict = pickle.load(file)

    # Classify files as wild-type or mutant
    for file_name in os.listdir(directory):
        if re.match(r'.+_wild_type\.pdb', file_name):
            wild_type_files[file_name.split('_wild_type')[0]] = file_name
        elif re.match(r'.+_mutant_.+\.pdb', file_name):
            base_name = file_name.split('_mutant_')[0]
            if base_name not in mutant_files:
                mutant_files[base_name] = []
            mutant_files[base_name].append(file_name)

    # 获取所有wild type PDB文件
    wild_pdbs = list(Path(directory).glob('*_wild_type.pdb'))
    # 预计算总突变体数量
    total_mutants = sum(len(list(wild_pdb.parent.glob(f'{wild_pdb.stem.replace("_wild_type", "")}_mutant_*.pdb')))
                        for wild_pdb in wild_pdbs)
    # 创建进度条
    if progress_bar:
        pbar = tqdm(total=total_mutants,
                    desc="Processing mutations",
                    unit="mutation",
                    ncols=100,
                    colour='green',
                    file=sys.stdout)
    # with tqdm(total=total_mutants,
    #           desc="Processing mutations",
    #           unit="mutation",
    #           ncols=100,
    #           colour='green',
    #           file=sys.stdout) as pbar:
        # Process each mutant file with its corresponding wild type
    for base_name, mutants in mutant_files.items():
        if base_name in wild_type_files:
            wild_type_file = wild_type_files[base_name]
            wild_type_structure = parser.get_structure(base_name + '_wild_type',
                                                       os.path.join(directory, wild_type_file))

            # Load PSSM and conservation scores for wild type
            wild_type_pssm = pssm_scores_dict.get(wild_type_file.replace('.pdb', '.pssm'), None)
            wild_type_conservation = load_conservation_scores(conservation_directory, wild_type_file)
            wild_type_graph = create_graph_from_structure(wild_type_structure, wild_type_pssm, wild_type_conservation)

            for mutant_file in mutants:
                mutant_structure = parser.get_structure(mutant_file, os.path.join(directory, mutant_file))

                # Load PSSM and conservation scores for mutant
                mutant_pssm = pssm_scores_dict.get(mutant_file.replace('.pdb', '.pssm'), None)
                mutant_conservation = load_conservation_scores(conservation_directory, mutant_file)

                mutant_graph = create_graph_from_structure(mutant_structure, mutant_pssm, mutant_conservation)

                graphs.append((wild_type_graph, mutant_graph, mutant_file))
                if progress_bar:
                    pbar.update(1)

    # 关闭进度条
    if progress_bar:
        pbar.close()

    return graphs

# -----------------------------------------------RNA-----------------------------------------------
# 添加到现有的utils.py文件中

# RNA核苷酸单字母代码与三字母代码的映射
rna_dict = {
    'A': 'ADE',  # 腺嘌呤
    'U': 'URI',  # 尿嘧啶
    'G': 'GUA',  # 鸟嘌呤
    'C': 'CYT',  # 胞嘧啶
}

# RNA核苷酸的one-hot编码
rna_one_hot = {
    'ADE': [1, 0, 0, 0],  # A
    'URI': [0, 1, 0, 0],  # U
    'GUA': [0, 0, 1, 0],  # G
    'CYT': [0, 0, 0, 1],  # C
}


def get_rna_center_coords_v1(structure):
    """
    获取RNA核苷酸中心点坐标
    使用碱基中的N1(对于嘧啶)/N9(对于嘌呤)原子作为中心点
    """
    center_coords = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                # 检查是否为核苷酸
                if residue.get_resname() in ['ADE', 'URI', 'GUA', 'CYT', 'A', 'U', 'G', 'C']:
                    # 对于腺嘌呤(A)和鸟嘌呤(G)，使用N9作为中心
                    if residue.get_resname() in ['ADE', 'GUA', 'A', 'G']:
                        if 'N9' in residue:
                            center_coords[(chain.id, residue.id)] = residue['N9'].coord
                    # 对于胞嘧啶(C)和尿嘧啶(U)，使用N1作为中心
                    elif residue.get_resname() in ['CYT', 'URI', 'C', 'U']:
                        if 'N1' in residue:
                            center_coords[(chain.id, residue.id)] = residue['N1'].coord
                    # 如果没有找到N9或N1，使用C1'作为替代
                    if (chain.id, residue.id) not in center_coords and "C1'" in residue:
                        center_coords[(chain.id, residue.id)] = residue["C1'"].coord
    return center_coords


def get_rna_center_coords_v2(structure):
    """
    获取RNA结构中所有核苷酸的中心坐标
    使用更精确的中心定义：嘌呤使用N9，嘧啶使用N1
    如果这些原子不存在，则使用C1'作为备选

    参数:
    - structure: BioPDB结构对象

    返回:
    - center_coords: 字典，键为(chain_id, residue_id)，值为中心坐标
    """
    from enhanced_rna_processing_debug import rna_dict, purine_types, pyrimidine_types

    center_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)

                    # 根据碱基类型选择中心原子
                    resname = residue.get_resname()
                    if resname in purine_types:
                        # 嘌呤使用N9作为中心
                        if 'N9' in residue:
                            center_coords[res_id] = residue['N9'].coord
                        elif "C1'" in residue:
                            center_coords[res_id] = residue["C1'"].coord
                        else:
                            # 如果都没有，使用所有原子的平均坐标
                            atoms = [atom.coord for atom in residue]
                            if atoms:
                                center_coords[res_id] = np.mean(atoms, axis=0)
                    elif resname in pyrimidine_types:
                        # 嘧啶使用N1作为中心
                        if 'N1' in residue:
                            center_coords[res_id] = residue['N1'].coord
                        elif "C1'" in residue:
                            center_coords[res_id] = residue["C1'"].coord
                        else:
                            # 如果都没有，使用所有原子的平均坐标
                            atoms = [atom.coord for atom in residue]
                            if atoms:
                                center_coords[res_id] = np.mean(atoms, axis=0)
                    else:
                        # 未知类型，使用所有原子的平均坐标
                        atoms = [atom.coord for atom in residue]
                        if atoms:
                            center_coords[res_id] = np.mean(atoms, axis=0)

    return center_coords


def get_rna_center_coords(structure):
    """
    获取RNA结构中所有核苷酸的中心坐标

    参数:
    - structure: BioPDB结构对象

    返回:
    - center_coords: 字典，键为(chain_id, residue_id)，值为中心坐标
    """
    # 导入RNA字典（假设从您的enhanced_rna_processing_debug.py中导入）
    from enhanced_rna_processing_debug import rna_dict

    center_coords = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                # 检查是否为RNA核苷酸
                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)

                    # 计算残基中心点坐标
                    atoms = [atom.coord for atom in residue]
                    if atoms:
                        center_coord = np.mean(atoms, axis=0)
                        center_coords[res_id] = center_coord

    return center_coords

def rna_contact_map(structure, threshold=8.0):
    """
    生成RNA接触图矩阵

    参数:
    structure: BioPython PDB结构对象
    threshold: 定义接触的距离阈值(埃)

    返回:
    contact_map: 核苷酸间接触的二维矩阵
    residue_info: 每个核苷酸的信息(链ID，残基ID，残基名)列表
    """
    center_coords = get_rna_center_coords(structure)
    residue_info = []

    # 收集所有核苷酸信息
    for (chain_id, res_id) in center_coords.keys():
        residue = structure[0][chain_id][res_id]
        residue_info.append((chain_id, res_id, residue.get_resname()))

    n_residues = len(residue_info)
    contact_map = np.zeros((n_residues, n_residues))

    # 计算接触图
    coords_list = list(center_coords.values())
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            distance = np.linalg.norm(coords_list[i] - coords_list[j])
            if distance <= threshold:
                contact_map[i, j] = 1
                contact_map[j, i] = 1  # 对称矩阵

    return contact_map, residue_info


def create_rna_graph_from_structure(structure, threshold=8.0, encoding_dim=ENCODING_DIM):
    """
    从RNA结构创建图对象，包含节点特征(核苷酸类型)和边特征(距离编码)

    参数:
    structure: BioPython PDB结构对象
    threshold: 定义接触的距离阈值(埃)
    encoding_dim: 位置编码的维度

    返回:
    graph: NetworkX图对象，包含RNA结构信息
    """
    center_coords = get_rna_center_coords(structure)
    graph = nx.Graph()

    # 添加节点和特征
    for (chain_id, residue_id), coord in center_coords.items():
        residue = structure[0][chain_id][residue_id]
        resname = residue.get_resname()
        # 将三字母代码转换为标准格式(如果需要)
        if resname in ['A', 'U', 'G', 'C']:
            resname = rna_dict.get(resname, resname)
        # 获取one-hot编码
        one_hot = np.array(rna_one_hot.get(resname, [0, 0, 0, 0]))
        # 添加节点
        graph.add_node((chain_id, residue_id),
                       feature=one_hot,
                       resname=resname)

    # 添加边
    for (chain_id1, res_id1), coord1 in center_coords.items():
        for (chain_id2, res_id2), coord2 in center_coords.items():
            if (chain_id1, res_id1) != (chain_id2, res_id2):
                distance = np.linalg.norm(coord1 - coord2)
                if distance < threshold:
                    edge_feature = positional_encoding(distance, encoding_dim)
                    graph.add_edge((chain_id1, res_id1),
                                   (chain_id2, res_id2),
                                   feature=edge_feature,
                                   distance=distance)

    return graph