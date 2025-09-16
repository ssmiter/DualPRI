# enhanced_rna_processing.py
# 增强的RNA处理模块，提供高级RNA图构建和特征提取功能

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import PDB
from Bio.PDB.vectors import calc_dihedral, calc_angle
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings
import networkx as nx

# 导入您的utils文件中的函数
from utils import positional_encoding

# 忽略BioPDB的警告
warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)

# 核苷酸one-hot编码字典
rna_dict = {
    'A': [1, 0, 0, 0, 0],  # 腺嘌呤
    'U': [0, 1, 0, 0, 0],  # 尿嘧啶
    'G': [0, 0, 1, 0, 0],  # 鸟嘌呤
    'C': [0, 0, 0, 1, 0],  # 胞嘧啶
    'N': [0, 0, 0, 0, 1],  # 未知或修饰的核苷酸
    'ADE': [1, 0, 0, 0, 0],
    'URI': [0, 1, 0, 0, 0],
    'GUA': [0, 0, 1, 0, 0],
    'CYT': [0, 0, 0, 1, 0],
    'I': [0, 0, 0, 0, 1],  # 肌苷
    'PSU': [0, 0, 0, 0, 1],  # 假尿苷
    'T': [0, 1, 0, 0, 0],  # 胸腺嘧啶 (在RNA中当作U处理)
}

# RNA二级结构类型编码
ss_type_encoding = {
    'stem': [1, 0, 0, 0, 0],  # 茎
    'hairpin': [0, 1, 0, 0, 0],  # 发夹环
    'internal_loop': [0, 0, 1, 0, 0],  # 内环
    'bulge': [0, 0, 0, 1, 0],  # 凸环
    'multi_branch': [0, 0, 0, 0, 1],  # 多分支环
    'unknown': [0, 0, 0, 0, 0]  # 未知结构
}

# 嘌呤和嘧啶类型
purine_types = ['A', 'G', 'ADE', 'GUA']
pyrimidine_types = ['U', 'C', 'URI', 'CYT', 'T']


def get_residue_atoms(residue):
    """获取残基中的关键原子坐标"""
    atom_coords = {}
    atom_list = ["P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

    # 碱基原子 - 根据是嘌呤还是嘧啶有所不同
    resname = residue.get_resname()
    if resname in ['A', 'G', 'ADE', 'GUA']:  # 嘌呤
        atom_list.extend(["N9", "C8", "N7", "C5", "C6", "N1", "C2"])
        if resname in ['A', 'ADE']:  # 腺嘌呤特有原子
            atom_list.extend(["N6", "C4", "N3"])
        elif resname in ['G', 'GUA']:  # 鸟嘌呤特有原子
            atom_list.extend(["O6", "C4", "N3", "N2"])
    elif resname in ['U', 'C', 'URI', 'CYT', 'T']:  # 嘧啶
        atom_list.extend(["N1", "C2", "N3", "C4", "C5", "C6"])
        if resname in ['U', 'URI', 'T']:  # 尿嘧啶特有原子
            atom_list.extend(["O4", "O2"])
        elif resname in ['C', 'CYT']:  # 胞嘧啶特有原子
            atom_list.extend(["N4", "O2"])

    for atom_name in atom_list:
        if atom_name in residue:
            atom_coords[atom_name] = residue[atom_name].coord

    return atom_coords


def calculate_rna_torsion_angles(residue, prev_residue=None, next_residue=None):
    """
    计算RNA骨架六个主要扭转角: α, β, γ, δ, ε, ζ
    以及糖环的构象伪转动角

    如果某些原子缺失或无法计算角度，则返回NaN
    """
    torsion_angles = {
        'alpha': np.nan,
        'beta': np.nan,
        'gamma': np.nan,
        'delta': np.nan,
        'epsilon': np.nan,
        'zeta': np.nan,
        'chi': np.nan,
        'pseudorotation': np.nan
    }

    # 获取当前残基的原子
    atoms = get_residue_atoms(residue)

    # 计算骨架扭转角 (如果有前一个和后一个残基)
    if prev_residue and "O3'" in atoms:
        prev_atoms = get_residue_atoms(prev_residue)

        # α: O3'(i-1)-P-O5'-C5'
        if all(atom in atoms for atom in ["P", "O5'", "C5'"]) and "O3'" in prev_atoms:
            try:
                torsion_angles['alpha'] = calc_dihedral(
                    PDB.Vector(prev_atoms["O3'"]),
                    PDB.Vector(atoms["P"]),
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"])
                )
            except:
                pass

        # β: P-O5'-C5'-C4'
        if all(atom in atoms for atom in ["P", "O5'", "C5'", "C4'"]):
            try:
                torsion_angles['beta'] = calc_dihedral(
                    PDB.Vector(atoms["P"]),
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"]),
                    PDB.Vector(atoms["C4'"])
                )
            except:
                pass

        # γ: O5'-C5'-C4'-C3'
        if all(atom in atoms for atom in ["O5'", "C5'", "C4'", "C3'"]):
            try:
                torsion_angles['gamma'] = calc_dihedral(
                    PDB.Vector(atoms["O5'"]),
                    PDB.Vector(atoms["C5'"]),
                    PDB.Vector(atoms["C4'"]),
                    PDB.Vector(atoms["C3'"])
                )
            except:
                pass

    # δ: C5'-C4'-C3'-O3'
    if all(atom in atoms for atom in ["C5'", "C4'", "C3'", "O3'"]):
        try:
            torsion_angles['delta'] = calc_dihedral(
                PDB.Vector(atoms["C5'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["O3'"])
            )
        except:
            pass

    # ε and ζ (需要后一个残基)
    if next_residue and "O3'" in atoms:
        next_atoms = get_residue_atoms(next_residue)

        # ε: C4'-C3'-O3'-P(i+1)
        if all(atom in atoms for atom in ["C4'", "C3'", "O3'"]) and "P" in next_atoms:
            try:
                torsion_angles['epsilon'] = calc_dihedral(
                    PDB.Vector(atoms["C4'"]),
                    PDB.Vector(atoms["C3'"]),
                    PDB.Vector(atoms["O3'"]),
                    PDB.Vector(next_atoms["P"])
                )
            except:
                pass

        # ζ: C3'-O3'-P(i+1)-O5'(i+1)
        if all(atom in atoms for atom in ["C3'", "O3'"]) and all(atom in next_atoms for atom in ["P", "O5'"]):
            try:
                torsion_angles['zeta'] = calc_dihedral(
                    PDB.Vector(atoms["C3'"]),
                    PDB.Vector(atoms["O3'"]),
                    PDB.Vector(next_atoms["P"]),
                    PDB.Vector(next_atoms["O5'"])
                )
            except:
                pass

    # χ: 糖基-碱基连接角
    resname = residue.get_resname()
    base_atom = "N9" if resname in purine_types else "N1"

    if all(atom in atoms for atom in ["O4'", "C1'", base_atom]):
        try:
            torsion_angles['chi'] = calc_dihedral(
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms[base_atom]),
                PDB.Vector(atoms["C2" if base_atom == "N1" else "C4"])
            )
        except:
            pass

    # 计算糖环的构象 (伪转动角)
    if all(atom in atoms for atom in ["C1'", "C2'", "C3'", "C4'", "O4'"]):
        try:
            # 计算五个内环扭转角
            v1 = calc_dihedral(
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"])
            )
            v2 = calc_dihedral(
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"])
            )
            v3 = calc_dihedral(
                PDB.Vector(atoms["C1'"]),
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"])
            )
            v4 = calc_dihedral(
                PDB.Vector(atoms["C2'"]),
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"])
            )
            v5 = calc_dihedral(
                PDB.Vector(atoms["C3'"]),
                PDB.Vector(atoms["C4'"]),
                PDB.Vector(atoms["O4'"]),
                PDB.Vector(atoms["C1'"])
            )

            # 使用Altona & Sundaralingam方程计算伪转动角P
            # P = arctan((v2+v5)-(v3+v4), 2*v1*(sin(36°)+sin(72°)))
            # 简化计算
            numerator = (v2 + v5) - (v3 + v4)
            denominator = 2 * v1 * (np.sin(np.radians(36)) + np.sin(np.radians(72)))
            p_angle = np.arctan2(numerator, denominator)
            torsion_angles['pseudorotation'] = np.degrees(p_angle) % 360

        except:
            pass

    return torsion_angles


def detect_ribose_puckering(residue):
    """
    确定核糖环构象: C3'-endo (北) 或 C2'-endo (南)

    返回one-hot编码 [C3'_endo, C2'_endo]
    """
    # 默认返回值 (未知构象)
    default_result = [0, 0]

    # 获取所需原子
    required_atoms = ["C1'", "C2'", "C3'", "C4'", "O4'"]
    atoms = {}

    for atom_name in required_atoms:
        if atom_name in residue:
            atoms[atom_name] = residue[atom_name].get_vector()
        else:
            return default_result  # 缺少必要原子

    try:
        # 计算伪转动角
        torsion_angles = calculate_rna_torsion_angles(residue)
        p_angle = torsion_angles['pseudorotation']

        if not np.isnan(p_angle):
            # 北方构象 (C3'-endo): 约 0° 到 36°
            # 南方构象 (C2'-endo): 约 144° 到 190°
            if 0 <= p_angle < 90 or 270 <= p_angle < 360:
                return [1, 0]  # C3'-endo
            elif 90 <= p_angle < 270:
                return [0, 1]  # C2'-endo

        # 备选方法：使用C2'到C4'-O4'-C1'平面的距离
        # 创建C4'-O4'-C1'平面
        plane_normal = (atoms["O4'"] - atoms["C4'"]).cross(atoms["C1'"] - atoms["O4'"]).normalized()

        # C2'到平面的距离
        c2_to_plane = (atoms["C2'"] - atoms["C4'"]).dot(plane_normal)

        if c2_to_plane > 0:
            return [0, 1]  # C2'-endo
        else:
            return [1, 0]  # C3'-endo

    except:
        # 如果计算失败，尝试使用简化方法
        try:
            # 检查C3'和C2'的位置关系
            c3_z = residue["C3'"].get_coord()[2]
            c2_z = residue["C2'"].get_coord()[2]

            if c3_z > c2_z:
                return [1, 0]  # C3'-endo
            else:
                return [0, 1]  # C2'-endo
        except:
            pass

    return default_result  # 返回默认未知构象


def predict_secondary_structure_type(residue_idx, chain_length, base_pairs):
    """
    根据碱基配对情况预测核苷酸所处的二级结构类型

    参数:
    - residue_idx: 当前残基的索引
    - chain_length: RNA链的总长度
    - base_pairs: 碱基配对列表，每个元素为 (i, j) 表示i和j位置的核苷酸配对

    返回:
    - one-hot编码的二级结构类型
    """
    # 默认返回未知结构
    default_result = ss_type_encoding['unknown']

    # 检查当前残基是否参与碱基配对
    paired_with = None
    try:
        for i, j in base_pairs:  # 注意这里解包两个值
            if i == residue_idx:
                paired_with = j
                break
            elif j == residue_idx:
                paired_with = i
                break
    except ValueError as e:
        print(f"警告: 二级结构预测时解包碱基配对出错: {str(e)}")
        return default_result

    # 如果不配对，可能是环状结构的一部分
    if paired_with is None:
        # 检查是否在已知茎段之间 (可能是环)
        in_loop = False
        for i, j in base_pairs:
            # 如果在一对碱基配对之间，可能是环的一部分
            if i < residue_idx < j or j < residue_idx < i:
                in_loop = True
                break

        if in_loop:
            # 确定环的类型需要更复杂的检查，这里简化处理
            # 实际应用中应更详细地分析环的类型
            return ss_type_encoding['hairpin']
        else:
            return default_result

    # 如果配对，检查是否是茎的一部分
    # 检查相邻碱基是否也配对
    adjacent_paired = False
    if residue_idx + 1 < chain_length and paired_with - 1 >= 0:
        for i, j in base_pairs:
            if (i == residue_idx + 1 and j == paired_with - 1) or (j == residue_idx + 1 and i == paired_with - 1):
                adjacent_paired = True
                break

    if adjacent_paired:
        return ss_type_encoding['stem']

    # 简化处理：可以进一步检查具体环类型，但需要更完整的二级结构信息
    # 这里返回默认的茎结构
    return ss_type_encoding['stem']


def detect_base_pairs(structure, distance_threshold=4.0, angle_threshold=35.0):
    """
    检测RNA结构中的碱基配对

    参数:
    - structure: BioPDB结构对象
    - distance_threshold: 氢键距离阈值
    - angle_threshold: 氢键角度阈值

    返回:
    - 碱基配对列表，每个元素为 (i, j, pair_type) 表示i和j位置的核苷酸配对及类型
    """
    base_pairs = []
    base_pairs_with_type = []  # 存储带有类型的三元组

    # 提取所有核苷酸
    nucleotides = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # 检查是否是RNA核苷酸
                if residue.get_resname() in list(rna_dict.keys()):
                    nucleotides.append(residue)

    # 枚举所有可能的配对
    for i, nt1 in enumerate(nucleotides):
        for j, nt2 in enumerate(nucleotides[i + 1:], i + 1):
            # 忽略相邻的核苷酸
            if abs(nt1.id[1] - nt2.id[1]) <= 1 and nt1.parent.id == nt2.parent.id:
                continue

            # 检查是否形成碱基配对
            pair_type = check_base_pairing(nt1, nt2, distance_threshold)
            if pair_type:
                base_pairs.append((i, j))  # 只存储索引对
                base_pairs_with_type.append((i, j, pair_type))  # 存储带类型的三元组

    return base_pairs, base_pairs_with_type  # 返回两种格式


def check_base_pairing(nt1, nt2, distance_threshold=4.0):
    """
    检查两个核苷酸之间是否形成碱基配对

    参数:
    - nt1, nt2: 核苷酸残基
    - distance_threshold: 氢键最大距离

    返回:
    - 配对类型: 'WC' (Watson-Crick), 'wobble', 或 None
    """
    # 获取两个核苷酸的名称
    name1 = nt1.get_resname()
    name2 = nt2.get_resname()

    # 简化版本：根据核苷酸类型和原子间距离判断
    if name1 in ['A', 'ADE'] and name2 in ['U', 'URI', 'T']:
        # A-U 配对
        if 'N1' in nt1 and 'N3' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['N3'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['U', 'URI', 'T'] and name2 in ['A', 'ADE']:
        # U-A 配对
        if 'N3' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['N3'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['G', 'GUA'] and name2 in ['C', 'CYT']:
        # G-C 配对
        if 'N1' in nt1 and 'N3' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['N3'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['C', 'CYT'] and name2 in ['G', 'GUA']:
        # C-G 配对
        if 'N3' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['N3'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'WC'

    elif name1 in ['G', 'GUA'] and name2 in ['U', 'URI', 'T']:
        # G-U wobble配对
        if 'N1' in nt1 and 'O2' in nt2:
            dist = np.linalg.norm(nt1['N1'].coord - nt2['O2'].coord)
            if dist <= distance_threshold:
                return 'wobble'

    elif name1 in ['U', 'URI', 'T'] and name2 in ['G', 'GUA']:
        # U-G wobble配对
        if 'O2' in nt1 and 'N1' in nt2:
            dist = np.linalg.norm(nt1['O2'].coord - nt2['N1'].coord)
            if dist <= distance_threshold:
                return 'wobble'

    # 其他可能的配对类型...

    return None


def calculate_solvent_accessibility(residue, probe_radius=1.4):
    """
    估计核苷酸的溶剂可及性

    参数:
    - residue: 核苷酸残基
    - probe_radius: 模拟水分子的探针半径

    返回:
    - 归一化的可及性分数 (0-1)
    """
    # 这是一个简化的计算，实际项目中应使用专门的软件如DSSP
    # 这里仅基于原子坐标估计暴露程度

    # 获取残基中所有原子
    atoms = [atom for atom in residue]
    if not atoms:
        return 0.5  # 默认中等可及性

    # 计算残基的"中心"位置
    center = np.mean([atom.coord for atom in atoms], axis=0)

    # 计算原子到中心的平均距离
    avg_distance = np.mean([np.linalg.norm(atom.coord - center) for atom in atoms])

    # 基于平均距离估计可及性 (越大越暴露)
    # 简化公式，实际应用需要更精确的计算
    accessibility = min(1.0, avg_distance / 10.0)

    return accessibility


def extract_rna_features(residue, prev_residue=None, next_residue=None, base_pairs=None, residue_idx=None,
                         chain_length=None):
    """
    提取单个RNA核苷酸的综合特征

    参数:
    - residue: 核苷酸残基
    - prev_residue, next_residue: 相邻残基，用于计算骨架扭转角
    - base_pairs: 碱基配对信息
    - residue_idx: 当前残基在链中的索引
    - chain_length: RNA链的总长度

    返回:
    - 特征向量，包含多种RNA特征
    """
    features = []

    # 1. 核苷酸类型 (one-hot编码)
    resname = residue.get_resname()
    nucleotide_type = rna_dict.get(resname, rna_dict['N'])
    features.extend(nucleotide_type)

    # 2. 核糖环构象 (C3'-endo vs C2'-endo)
    puckering = detect_ribose_puckering(residue)
    features.extend(puckering)

    # 3. 二级结构类型
    if base_pairs is not None and residue_idx is not None and chain_length is not None:
        try:
            ss_feature = predict_secondary_structure_type(residue_idx, chain_length, base_pairs)
        except Exception as e:
            print(f"警告: 预测二级结构类型失败: {str(e)}")
            ss_feature = ss_type_encoding['unknown']
    else:
        ss_feature = ss_type_encoding['unknown']
    features.extend(ss_feature)

    # 4. 溶剂可及性
    accessibility = calculate_solvent_accessibility(residue)
    features.append(accessibility)

    # 5. 骨架扭转角特征
    torsion_angles = calculate_rna_torsion_angles(residue, prev_residue, next_residue)

    # 归一化角度 (将角度转换为-1到1之间)
    normalized_angles = []
    for angle_name in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']:
        angle = torsion_angles.get(angle_name, np.nan)
        if not np.isnan(angle):
            # 转换为弧度并归一化
            norm_angle = np.cos(np.radians(angle))
            normalized_angles.append(norm_angle)
            # 添加正弦值捕获周期性
            norm_angle_sin = np.sin(np.radians(angle))
            normalized_angles.append(norm_angle_sin)
        else:
            normalized_angles.extend([0, 0])  # 如果无法计算，使用零值

    features.extend(normalized_angles)

    # 6. 伪转动角的正弦和余弦
    pseudo_angle = torsion_angles.get('pseudorotation', np.nan)
    if not np.isnan(pseudo_angle):
        features.append(np.cos(np.radians(pseudo_angle)))
        features.append(np.sin(np.radians(pseudo_angle)))
    else:
        features.extend([0, 0])

    # 7. 是否参与碱基配对
    if base_pairs is not None and residue_idx is not None:
        is_paired = 0
        try:
            for i, j in base_pairs:  # 注意这里解包两个值
                if i == residue_idx or j == residue_idx:
                    is_paired = 1
                    break
        except ValueError as e:
            print(f"警告: 检查碱基配对时出错: {str(e)}")
            is_paired = 0
        features.append(is_paired)
    else:
        features.append(0)

    # 8. 化学特性 (简化)
    # 嘌呤/嘧啶
    is_purine = 1 if resname in purine_types else 0
    is_pyrimidine = 1 if resname in pyrimidine_types else 0
    features.extend([is_purine, is_pyrimidine])

    # 添加电荷相关特性
    # RNA磷酸骨架通常带负电荷
    features.append(1 if "P" in residue else 0)

    return np.array(features)


def create_rna_graph_from_structure(structure, threshold=8.0, include_base_pairs=True):
    """
    根据RNA结构创建图表示

    参数:
    - structure: BioPDB结构对象
    - threshold: 接触距离阈值
    - include_base_pairs: 是否将碱基配对信息纳入图中

    返回:
    - G: NetworkX图对象，节点包含特征，边包含距离编码
    """
    import networkx as nx

    # 创建图
    G = nx.Graph()

    # 收集所有RNA残基并添加节点
    residues = []
    residue_coords = {}  # 用于计算中心点坐标

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    res_id = (chain.id, residue.id)
                    residues.append(residue)

                    # 计算残基中心点坐标
                    atoms = [atom.coord for atom in residue]
                    if atoms:
                        center_coord = np.mean(atoms, axis=0)
                        residue_coords[res_id] = center_coord

                    # 添加到图中，特征暂时为None
                    G.add_node(res_id, residue=residue)

    # 获取碱基配对信息
    if include_base_pairs:
        try:
            base_pairs, base_pairs_with_type = detect_base_pairs(structure)
        except Exception as e:
            print(f"警告: 无法检测碱基配对: {str(e)}")
            base_pairs = []
            base_pairs_with_type = []
    else:
        base_pairs = []
        base_pairs_with_type = []

    # 为每个节点提取特征
    chain_length = len(residues)
    for i, residue in enumerate(residues):
        res_id = (residue.get_parent().id, residue.id)

        # 获取前后残基用于计算扭转角
        prev_residue = residues[i - 1] if i > 0 else None
        next_residue = residues[i + 1] if i < len(residues) - 1 else None

        # 提取特征
        features = extract_rna_features(
            residue,
            prev_residue,
            next_residue,
            base_pairs=base_pairs,  # 这里只传递索引对
            residue_idx=i,
            chain_length=chain_length
        )

        # 更新节点特征
        G.nodes[res_id]['feature'] = features

    # 添加边
    for i, res_i in enumerate(residues):
        id_i = (res_i.get_parent().id, res_i.id)
        coord_i = residue_coords.get(id_i)

        if coord_i is None:
            continue

        for j, res_j in enumerate(residues[i + 1:], i + 1):
            id_j = (res_j.get_parent().id, res_j.id)
            coord_j = residue_coords.get(id_j)

            if coord_j is None:
                continue

            # 计算两个残基中心点之间的距离
            distance = np.linalg.norm(coord_i - coord_j)

            # 如果距离小于阈值，添加边
            if distance <= threshold:
                # 使用positional_encoding作为边特征
                edge_feature = positional_encoding(distance)
                G.add_edge(id_i, id_j, distance=distance, feature=edge_feature)

    # 添加碱基配对边
    for i, j, pair_type in base_pairs_with_type:
        id_i = (residues[i].get_parent().id, residues[i].id)
        id_j = (residues[j].get_parent().id, residues[j].id)

        # 可能已经添加过这条边
        if G.has_edge(id_i, id_j):
            # 添加碱基配对信息到现有边
            G[id_i][id_j]['base_pair'] = pair_type
        else:
            # 碱基配对的距离可能超过阈值，仍添加边
            # 使用positional_encoding作为边特征
            distance = np.linalg.norm(residue_coords[id_i] - residue_coords[id_j])
            edge_feature = positional_encoding(distance)
            G.add_edge(id_i, id_j, distance=distance, feature=edge_feature, base_pair=pair_type)

    return G


def visualize_contact_map(contact_map, residue_info, output_file=None):
    """可视化RNA接触图"""
    plt.figure(figsize=(10, 8))
    plt.imshow(contact_map, cmap='viridis')
    plt.colorbar(label='Contact')

    # 添加标签
    n_residues = len(residue_info)
    step = max(1, n_residues // 20)  # 控制标签数量
    plt.xticks(np.arange(0, n_residues, step),
               [f"{info[0]}:{info[1][1]}" for info in residue_info[::step]],
               rotation=90)
    plt.yticks(np.arange(0, n_residues, step),
               [f"{info[0]}:{info[1][1]}" for info in residue_info[::step]])

    plt.title(f'RNA Contact Map (threshold: 8Å)')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Contact map saved to {output_file}")
    else:
        plt.show()

    plt.close()

def visualize_rna_graph(G, output_file=None):
    """
    可视化RNA图结构

    参数:
    - G: NetworkX图对象
    - output_file: 输出文件路径
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # 创建可视化
    plt.figure(figsize=(12, 10))

    # 使用spring_layout布局
    pos = nx.spring_layout(G, seed=42)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=50)

    # 绘制边
    # 普通接触边
    normal_edges = [(u, v) for u, v in G.edges() if 'base_pair' not in G[u][v]]
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1, alpha=0.5)

    # 碱基配对边 (如果有)
    base_pair_edges = [(u, v) for u, v in G.edges() if 'base_pair' in G[u][v]]
    if base_pair_edges:
        nx.draw_networkx_edges(G, pos, edgelist=base_pair_edges, width=2,
                               edge_color='red', style='dashed')

    # 添加标签
    labels = {node: f"{node[0]}:{node[1][1]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("RNA Structure Graph Representation")
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_file}")
    else:
        plt.show()

    plt.close()

def process_rna_pdbs(directory, output_dir='./rna_analysis'):
    """
    处理目录中的RNA PDB文件，生成接触图和图结构

    参数:
    - directory: RNA PDB文件所在目录
    - output_dir: 输出结果的目录

    返回:
    - results: 包含处理结果的字典
    """
    parser = PDB.PDBParser(QUIET=True)
    results = {}

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有RNA PDB文件
    rna_pdbs = list(Path(directory).glob('*_nucleic.pdb'))

    # 创建进度条
    with tqdm(total=len(rna_pdbs),
              desc="Processing RNA structures",
              unit="file",
              ncols=100,
              colour='blue',
              file=sys.stdout) as pbar:

        for pdb_path in rna_pdbs:
            pdb_id = pdb_path.stem.split('_')[0]

            # 检查文件大小
            file_size_mb = pdb_path.stat().st_size / (1024 * 1024)  # 转换为MB
            if file_size_mb > 1:
                print(f"跳过大型文件 {pdb_id} (大小: {file_size_mb:.2f} MB > {1} MB)")
                pbar.update(1)
                continue
            try:
                # 解析PDB结构
                structure = parser.get_structure(pdb_id, pdb_path)

                # 生成接触图
                contact_map, residue_info = rna_contact_map(structure, threshold=8.0)

                # 生成图结构 - 使用异常处理
                try:
                    graph = create_rna_graph_from_structure(structure, threshold=8.0)
                except Exception as e:
                    print(f"警告: 为 {pdb_id} 创建图结构失败: {str(e)}")
                    # 创建一个只有节点没有边的简单图
                    graph = nx.Graph()
                    for idx, info in enumerate(residue_info):
                        graph.add_node(info, feature=np.zeros(30))  # 使用默认特征

                # 保存结果
                results[pdb_id] = {
                    'contact_map': contact_map,
                    'residue_info': residue_info,
                    'graph': graph,
                    'pdb_path': str(pdb_path)
                }

                # 可视化接触图
                visualize_contact_map(
                    contact_map,
                    residue_info,
                    output_file=os.path.join(output_dir, f"{pdb_id}_contact_map.png")
                )

                # 可视化图结构
                visualize_rna_graph(
                    graph,
                    output_file=os.path.join(output_dir, f"{pdb_id}_graph.png")
                )

                # 保存详细的节点特征信息
                node_features = {node: data.get('feature') for node, data in graph.nodes(data=True)}
                feature_df = pd.DataFrame(
                    {f"{node[0]}:{node[1][1]}": feat for node, feat in node_features.items() if feat is not None}
                ).T

                # 特征名称
                feature_names = [
                    'A', 'U', 'G', 'C', 'N',  # 核苷酸类型
                    'C3_endo', 'C2_endo',  # 核糖环构象
                    'stem', 'hairpin', 'internal_loop', 'bulge', 'multi_branch',  # 二级结构
                    'accessibility',  # 溶剂可及性
                    # 接下来是扭转角的余弦和正弦值
                    'alpha_cos', 'alpha_sin',
                    'beta_cos', 'beta_sin',
                    'gamma_cos', 'gamma_sin',
                    'delta_cos', 'delta_sin',
                    'epsilon_cos', 'epsilon_sin',
                    'zeta_cos', 'zeta_sin',
                    'chi_cos', 'chi_sin',
                    'pseudo_cos', 'pseudo_sin',  # 伪转动角
                    'is_paired',  # 是否配对
                    'is_purine', 'is_pyrimidine',  # 化学特性
                    'has_phosphate'  # 磷酸基团
                ]

                if feature_df.shape[1] == len(feature_names):
                    feature_df.columns = feature_names

                feature_df.to_csv(os.path.join(output_dir, f"{pdb_id}_node_features.csv"))

            except Exception as e:
                print(f"Error processing {pdb_id}: {e}")
                pbar.update(1)
                continue

            pbar.update(1)

    # 保存所有处理结果
    with open(os.path.join(output_dir, 'rna_structures_data.pkl'), 'wb') as f:
        # 只保存可序列化的部分
        serializable_results = {}
        for pdb_id, data in results.items():
            serializable_results[pdb_id] = {
                'contact_map': data['contact_map'],
                'residue_info': data['residue_info'],
                'pdb_path': data['pdb_path']
                # NetworkX图不容易序列化，略过
            }
        pickle.dump(serializable_results, f)

    return results

def analyze_rna_features(results):
    """
    分析RNA特征，生成统计信息

    参数:
    - results: 处理结果字典

    返回:
    - 分析统计数据
    """
    print("\nRNA Feature Analysis Summary:")
    print("=" * 50)

    all_features = []
    feature_info = []  # 存储特征信息

    # 收集所有特征及其信息
    for pdb_id, data in results.items():
        graph = data['graph']
        for node, node_data in graph.nodes(data=True):
            if 'feature' in node_data and node_data['feature'] is not None:
                feat = node_data['feature']
                all_features.append(feat)
                feature_info.append({
                    'pdb_id': pdb_id,
                    'node': node,
                    'dim': len(feat)
                })

    # 检查不同维度
    dims = [info['dim'] for info in feature_info]
    unique_dims = set(dims)
    print(f"发现特征维度: {unique_dims}")

    if len(unique_dims) > 1:
        print("\n异常维度的特征:")
        most_common_dim = max(set(dims), key=dims.count)
        print(f"最常见的维度: {most_common_dim}")

        # 打印前5个异常维度的信息
        abnormal_info = [info for info in feature_info if info['dim'] != most_common_dim]
        for i, info in enumerate(abnormal_info[:5]):
            print(f"  {i + 1}. PDB: {info['pdb_id']}, 节点: {info['node']}, 维度: {info['dim']}")

        if len(abnormal_info) > 5:
            print(f"  ... 还有 {len(abnormal_info) - 5} 个异常维度的特征")

    # 只使用最常见维度的特征
    most_common_dim = max(set(dims), key=dims.count)
    filtered_features = [feat for i, feat in enumerate(all_features) if dims[i] == most_common_dim]

    print(f"\n筛选后特征数量: {len(filtered_features)}/{len(all_features)}")

    # 转换为NumPy数组并计算统计量
    if filtered_features:
        features_array = np.vstack(filtered_features)

        # 计算统计量
        feature_means = np.mean(features_array, axis=0)
        feature_stds = np.std(features_array, axis=0)
        feature_mins = np.min(features_array, axis=0)
        feature_maxs = np.max(features_array, axis=0)

        # 特征名称 (调整为匹配实际维度)
        feature_names = [
            'A', 'U', 'G', 'C', 'N',  # 5
            'C3_endo', 'C2_endo',  # 2
            'stem', 'hairpin', 'internal_loop', 'bulge', 'multi_branch',  # 5
            'accessibility',  # 1
            'alpha_cos', 'alpha_sin',  # 2
            'beta_cos', 'beta_sin',  # 2
            'gamma_cos', 'gamma_sin',  # 2
            'delta_cos', 'delta_sin',  # 2
            'epsilon_cos', 'epsilon_sin',  # 2
            'zeta_cos', 'zeta_sin',  # 2
            'chi_cos', 'chi_sin',  # 2
            'pseudo_cos', 'pseudo_sin',  # 2
            'is_paired',  # 1
            'is_purine', 'is_pyrimidine',  # 2
            'has_phosphate'  # 1
        ]

        # 调整特征名称以匹配实际维度
        if len(feature_names) > most_common_dim:
            feature_names = feature_names[:most_common_dim]
        elif len(feature_names) < most_common_dim:
            for i in range(len(feature_names), most_common_dim):
                feature_names.append(f'feature_{i}')

        # 创建统计数据框
        stats_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean': feature_means,
            'Std': feature_stds,
            'Min': feature_mins,
            'Max': feature_maxs
        })

        print("\nFeature Statistics:")
        print(stats_df)

        # 保存统计数据
        output_dir = './rna_analysis'
        os.makedirs(output_dir, exist_ok=True)
        stats_df.to_csv(os.path.join(output_dir, 'rna_feature_stats.csv'), index=False)

        return stats_df
    else:
        print("No valid features found for analysis.")
        return None

def rna_contact_map(structure, threshold=8.0):
    """
    构建RNA接触图

    参数:
    - structure: BioPDB结构对象
    - threshold: 接触距离阈值

    返回:
    - contact_map: 二维接触矩阵
    - residue_info: 残基信息列表
    """
    # 收集所有RNA残基
    residues = []
    residue_info = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in list(rna_dict.keys()):
                    residues.append(residue)
                    residue_info.append((chain.id, residue.id))

    n_residues = len(residues)
    contact_map = np.zeros((n_residues, n_residues))

    # 计算接触图
    for i in range(n_residues):
        for j in range(n_residues):
            if i != j:  # 不考虑自身接触
                # 找到最小原子间距离
                min_distance = float('inf')

                for atom1 in residues[i]:
                    for atom2 in residues[j]:
                        distance = np.linalg.norm(atom1.coord - atom2.coord)
                        if distance < min_distance:
                            min_distance = distance

                #如果距离小于阈值，设置为1
                if min_distance <= threshold:
                    contact_map[i, j] = 1

    return contact_map, residue_info


if __name__ == "__main__":
    # 处理指定目录中的RNA PDB文件
    directory = "./processed_pdbs/nucleic_chains"
    output_dir = "./rna_analysis"

    # 确保目录存在
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理RNA结构
    results = process_rna_pdbs(directory, output_dir)

    # 分析特征
    stats_df = analyze_rna_features(results)

    print("\nRNA processing completed successfully!")
    print(f"Results saved to {output_dir}")