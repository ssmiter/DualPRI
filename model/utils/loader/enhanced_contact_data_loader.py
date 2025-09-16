"""
Enhanced protein-RNA interaction data loader with both node-level and block-level contact features
直接在数据加载阶段合并特征
"""
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import pickle
import numpy as np
import random
import os
from scipy.spatial.distance import cdist
import time

# 从原有代码导入
# from model.utils.loader.contact_data_loader import collate_protein_rna_triple

# 尝试从配置文件导入
try:
    from config import SHUFFLE, NUM_WORKERS, PREFETCH_FACTOR, DEFAULT_CHUNK_SIZE, DEFAULT_CONTACT_THRESHOLDS, DEFAULT_ESM2_CACHE_DIR, ESM2_FEATURE_DIM, ESM2_FEATURE_TYPES
except ImportError:
    SHUFFLE = True
    NUM_WORKERS = 0
    PREFETCH_FACTOR = 1
    DEFAULT_ESM2_CACHE_DIR = "./esm2_features"
    ESM2_FEATURE_DIM = 1280
    ESM2_FEATURE_TYPES = [4, 5, 6, 7]

# from utils import is_esm2_feature_type, get_feature_type_name, get_feature_type_details

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

class EnhancedProteinRNADataLoader:
    """增强的蛋白质-RNA相互作用数据加载器，支持节点级和块级接触特征，直接合并特征"""

    def __init__(self, data_path, batch_size=1, val_ratio=0.15, test_ratio=0.15,
                 add_reverse=True, seed=42, shuffle=True, chunk_size=DEFAULT_CHUNK_SIZE,
                 contact_thresholds=DEFAULT_CONTACT_THRESHOLDS, cache_dir="./contact_cache",
                 use_cached_contacts=True, num_workers=NUM_WORKERS, force_recompute=False,
                 prefetch_factor=PREFETCH_FACTOR, compute_block_features=True, compute_node_features=True,
                 feature_type=3, split_strategy=None, train_ratio=None,
                 esm2_cache_dir=DEFAULT_ESM2_CACHE_DIR, check_esm2_features=True):
        """
        Initialize data loader

    Parameters:
        data_path: Dataset path
        batch_size: Batch size
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        add_reverse: Whether to add reverse mutation samples
        seed: Random seed
        shuffle: Whether to shuffle data
        chunk_size: Block size
        contact_thresholds: Contact thresholds list
        cache_dir: Cache directory, None means no caching
        use_cached_contacts: Whether to use cached contact information
        force_recompute: Whether to force recompute contact information
        num_workers: Number of worker processes for DataLoader
        prefetch_factor: Prefetch factor for DataLoader
        compute_block_features: Whether to compute block-level contact features
        compute_node_features: Whether to compute node-level contact features
        feature_type: 多尺度特征类型 (0=无特征, 1=仅分布特征, 2=仅强度特征, 3=完整特征)
        split_strategy: Dataset splitting strategy (None, 'pdb_limited', 'directional', etc.)
        train_ratio: Training set ratio (only used with 'train_ratio' strategy)
        """
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.contact_thresholds = contact_thresholds
        self.cache_dir = cache_dir
        self.use_cached_contacts = use_cached_contacts
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.force_recompute = force_recompute
        self.compute_block_features = compute_block_features
        self.compute_node_features = compute_node_features
        self.feature_type = feature_type
        self.merge_features = feature_type > 0  # 只要feature_type > 0就合并特征
        self.train_ratio = train_ratio
        self.esm2_cache_dir = esm2_cache_dir
        self.check_esm2_features = check_esm2_features
        self.use_esm2_features = is_esm2_feature_type(feature_type)

        # 🔥 显示特征类型信息
        print(f"特征类型: {get_feature_type_name(feature_type)} (feature_type={feature_type})")
        if self.use_esm2_features:
            print(f"ESM2特征目录: {self.esm2_cache_dir}")
            # 检查ESM2特征可用性
            if not os.path.exists(self.esm2_cache_dir):
                raise FileNotFoundError(f"ESM2特征目录不存在: {self.esm2_cache_dir}")

            esm2_files = [f for f in os.listdir(self.esm2_cache_dir) if f.endswith('_esm2.pt')]
            print(f"发现 {len(esm2_files)} 个ESM2特征文件")

            if len(esm2_files) == 0:
                raise FileNotFoundError(f"ESM2特征目录中没有找到特征文件: {self.esm2_cache_dir}")

        # 创建缓存目录
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Cache directory: {self.cache_dir}")

            # 如果强制重新计算，则清除缓存
            if force_recompute:
                self.clear_contact_cache()
                print(f"已强制清除接触信息缓存，将使用新的阈值: {self.contact_thresholds}")

        # 设置随机种子
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # 加载数据
        start_time = time.time()
        print(f"Loading dataset: {data_path}")
        self.raw_data = self.load_data(data_path)
        print(f"Loaded {len(self.raw_data)} samples in {time.time() - start_time:.2f}s")

        # 转换为PyG格式，包含节点级和块级接触信息
        start_time = time.time()
        self.data_list = self.convert_to_pyg_format(self.raw_data)
        print(f"Converted data to PyG format in {time.time() - start_time:.2f}s")

        # 添加反向突变样本
        if add_reverse:
            start_time = time.time()
            self.add_reverse_mutations()
            print(f"Added reverse mutations in {time.time() - start_time:.2f}s")
            print(f"Total samples after adding reverse mutations: {len(self.data_list)}")

        # 拆分数据集
        start_time = time.time()
        self.train_data, self.val_data, self.test_data = self.split_dataset(
            val_ratio, test_ratio, strategy=split_strategy, train_ratio=train_ratio
        )
        print(f"Split dataset in {time.time() - start_time:.2f}s")

        # 创建数据加载器
        start_time = time.time()
        self.train_loader, self.val_loader, self.test_loader = self.create_data_loaders()
        print(f"Created data loaders in {time.time() - start_time:.2f}s")
        print(f"Initialization complete, ready for training")
        # 在初始化类后添加以下日志
        print(f"Feature merging: {'Enabled' if self.merge_features else 'Disabled'}")
        print(f"Node features: {'Enabled' if compute_node_features else 'Disabled'}")
        print(f"Block features: {'Enabled' if compute_block_features else 'Disabled'}")
        print(f"Using thresholds: {contact_thresholds}")

    def load_data(self, data_path):
        """加载Pickle格式数据集"""
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    def compute_node_contacts(self, protein_coords, rna_coords, thresholds=None):
        """
        计算节点级多尺度接触特征 - 分段区间计算方法

        参数:
        - protein_coords: 蛋白质坐标 [num_residues, 3]
        - rna_coords: RNA坐标 [num_nucleotides, 3]
        - thresholds: 距离阈值列表

        返回:
        - protein_contact_features: 蛋白质节点的接触特征 [num_residues, num_classes]
        - protein_contact_intensity: 蛋白质节点的接触强度 [num_residues, 1]
        """
        # 使用默认阈值
        if thresholds is None:
            thresholds = self.contact_thresholds

        # 确保输入是torch tensor
        if not isinstance(protein_coords, torch.Tensor):
            protein_coords = torch.tensor(protein_coords, dtype=torch.float)
        if not isinstance(rna_coords, torch.Tensor):
            rna_coords = torch.tensor(rna_coords, dtype=torch.float)

        # 确保在CPU上
        if protein_coords.is_cuda:
            protein_coords = protein_coords.cpu()
        if rna_coords.is_cuda:
            rna_coords = rna_coords.cpu()

        # 计算距离矩阵
        dist_matrix = torch.cdist(protein_coords, rna_coords)

        # 初始化
        num_residues = len(protein_coords)
        num_nucleotides = len(rna_coords)
        num_classes = len(thresholds) + 1

        protein_contact_features = torch.zeros((num_residues, num_classes))

        # 计算接触强度（基于最小距离）
        min_distances, _ = torch.min(dist_matrix, dim=1)
        protein_contact_intensity = 1.0 / (1.0 + torch.exp((min_distances - 8.0) / 2.0)).unsqueeze(1)

        # 计算分段区间接触特征
        # 第一个区间 < thresholds[0]
        mask = (dist_matrix < thresholds[0])
        protein_contact_features[:, 0] = mask.float().sum(dim=1) / num_nucleotides

        # 中间区间 thresholds[j-1] <= dist < thresholds[j]
        for j in range(1, len(thresholds)):
            mask = (dist_matrix >= thresholds[j-1]) & (dist_matrix < thresholds[j])
            protein_contact_features[:, j] = mask.float().sum(dim=1) / num_nucleotides

        # 最后一个区间 >= 最大阈值
        mask = (dist_matrix >= thresholds[-1])
        protein_contact_features[:, -1] = mask.float().sum(dim=1) / num_nucleotides

        return protein_contact_features, protein_contact_intensity

    def compute_block_contacts(self, protein_coords, rna_coords, chunk_size=None, thresholds=None):
        """
        计算块级接触分布和强度

        参数:
        protein_coords: 蛋白质坐标 [num_residues, 3]
        rna_coords: RNA坐标 [num_nucleotides, 3]
        chunk_size: 块大小
        thresholds: 距离阈值列表

        返回:
        block_distributions: 块级接触分布 [num_blocks, num_classes]，其中num_classes = len(thresholds) + 1
        block_intensities: 块级接触强度 [num_blocks, 1]
        """
        # 使用默认参数
        if chunk_size is None:
            chunk_size = self.chunk_size
        if thresholds is None:
            thresholds = self.contact_thresholds

        # 计算距离矩阵
        if isinstance(protein_coords, torch.Tensor):
            if protein_coords.is_cuda:
                protein_coords = protein_coords.cpu()
            if isinstance(rna_coords, torch.Tensor) and rna_coords.is_cuda:
                rna_coords = rna_coords.cpu()

            dist_matrix = torch.cdist(protein_coords, rna_coords)
        else:
            dist_matrix = torch.tensor(cdist(protein_coords, rna_coords))

        # 初始化块级信息
        num_residues = len(protein_coords)
        num_blocks = (num_residues + chunk_size - 1) // chunk_size

        # 类别数 = 阈值数 + 1
        num_classes = len(thresholds) + 1

        block_distributions = torch.zeros((num_blocks, num_classes))
        block_intensities = torch.zeros((num_blocks, 1))

        # 计算块级接触
        for i in range(num_blocks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_residues)

            if start_idx >= num_residues:
                break

            # 提取该块的距离
            block_dists = dist_matrix[start_idx:end_idx]

            # 计算接触分布（距离直方图）
            dist_hist = torch.zeros(num_classes)

            # 处理前N个类别（N = len(thresholds)）
            for j in range(len(thresholds)):
                if j == 0:
                    # 第一个区间: < thresholds[0]
                    mask = (block_dists < thresholds[0])
                else:
                    # 中间区间: thresholds[j-1] <= dist < thresholds[j]
                    mask = (block_dists >= thresholds[j - 1]) & (block_dists < thresholds[j])

                dist_hist[j] = mask.float().sum()

            # 最后一个类别: >= 最大阈值
            mask = (block_dists >= thresholds[-1])
            dist_hist[-1] = mask.float().sum()

            # 归一化分布
            total_pairs = dist_hist.sum()
            if total_pairs > 0:
                dist_hist = dist_hist / total_pairs

            # 计算接触强度 (基于最小距离)
            min_dist = block_dists.min().item()

            # 调整接触强度计算，使其对大范围距离更敏感
            # 8.0是拐点，小于8.0的距离接触强度接近1，大于的接近0
            intensity = 1.0 / (1.0 + np.exp((min_dist - 8.0) / 2.0))

            # 存储结果
            block_distributions[i] = dist_hist
            block_intensities[i] = intensity

        return block_distributions, block_intensities

    def get_block_contact_cache_path(self, pdb_id, chain_id, mutation=None, is_wild=True):
        """获取块级接触信息缓存路径"""
        if self.cache_dir is None:
            return None

        # 创建阈值标识符
        thresholds_str = "_".join([str(t) for t in self.contact_thresholds])

        protein_type = "wild" if is_wild else "mutant"
        if mutation:
            return os.path.join(self.cache_dir,
                                f"{pdb_id}_{chain_id}_{protein_type}_{mutation}_thresh{thresholds_str}_block_contacts.pt")
        else:
            return os.path.join(self.cache_dir,
                                f"{pdb_id}_{chain_id}_{protein_type}_thresh{thresholds_str}_block_contacts.pt")

    def get_node_contact_cache_path(self, pdb_id, chain_id, mutation=None, is_wild=True):
        """获取节点级接触特征缓存路径"""
        if self.cache_dir is None:
            return None

        # 创建阈值标识符
        thresholds_str = "_".join([str(t) for t in self.contact_thresholds])

        protein_type = "wild" if is_wild else "mutant"
        if mutation:
            return os.path.join(self.cache_dir,
                           f"{pdb_id}_{chain_id}_{protein_type}_{mutation}_thresh{thresholds_str}_node_contacts.pt")
        else:
            return os.path.join(self.cache_dir,
                           f"{pdb_id}_{chain_id}_{protein_type}_thresh{thresholds_str}_node_contacts.pt")

    def clear_contact_cache(self):
        """清除接触信息缓存"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            cache_files = os.listdir(self.cache_dir)
            cleared_count = 0
            for file in cache_files:
                if file.endswith("_contacts.pt"):
                    os.remove(os.path.join(self.cache_dir, file))
                    cleared_count += 1
            print(f"已清除 {cleared_count} 个缓存文件")

    def _check_thresholds_match(self, cached_thresholds, current_thresholds):
        """检查缓存的阈值与当前阈值是否匹配"""
        if cached_thresholds is None:
            return False

        # 转换为列表进行比较
        if isinstance(cached_thresholds, torch.Tensor):
            cached_thresholds = cached_thresholds.tolist()
        if isinstance(current_thresholds, torch.Tensor):
            current_thresholds = current_thresholds.tolist()

        # 处理可能的嵌套列表
        if isinstance(cached_thresholds, list) and len(cached_thresholds) > 0:
            if isinstance(cached_thresholds[0], list):
                cached_thresholds = cached_thresholds[0]

        if isinstance(current_thresholds, list) and len(current_thresholds) > 0:
            if isinstance(current_thresholds[0], list):
                current_thresholds = current_thresholds[0]

        # 检查数组长度和内容是否匹配
        return (len(cached_thresholds) == len(current_thresholds) and
                all(abs(a - b) < 1e-5 for a, b in zip(cached_thresholds, current_thresholds)))

    def convert_to_pyg_format(self, dataset):
        """
        转换数据为PyG格式，并计算节点级和块级接触信息
        已集成ESM2特征支持
        """
        pyg_data_list = []
        cached_node_count = 0
        computed_node_count = 0
        cached_block_count = 0
        computed_block_count = 0

        # 统计信息
        esm2_loaded_count = 0
        esm2_missing_count = 0
        esm2_dimension_mismatch = 0
        for i, item in enumerate(dataset):
            try:
                # 检查必要的数据字段
                if not all(key in item and item[key] is not None for key in
                           ['wild_graph', 'mutant_graph', 'rna_graph', 'metadata']):
                    print(f"Skipping sample {i}: Missing essential data fields")
                    continue

                # 检查各个图的特征
                for graph_key in ['wild_graph', 'mutant_graph', 'rna_graph']:
                    if not all(key in item[graph_key] and item[graph_key][key] is not None
                               for key in ['features', 'edge_index', 'edge_attr', 'coords']):
                        print(f"Skipping sample {i}: Missing data in {graph_key}")
                        continue

                # 提取元数据
                metadata = item['metadata']
                pdb_id = metadata.get('pdb_id', 'unknown')
                chain_id = metadata.get('chain_id', 'unknown')
                mutation = metadata.get('mutation', 'unknown')
                ddg = metadata.get('ddg')
                mutation_pos = metadata.get('mutation_pos')

                if ddg is None:
                    print(f"Skipping sample {i}: Missing DDG value")
                    continue

                # Wild-type protein data
                wild_data = Data(
                    # 基本特征
                    x=torch.tensor(item['wild_graph']['features'], dtype=torch.float),
                    edge_index=torch.tensor(item['wild_graph']['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(item['wild_graph']['edge_attr'], dtype=torch.float),
                    pos=torch.tensor(item['wild_graph']['coords'], dtype=torch.float),
                    batch=torch.zeros(len(item['wild_graph']['features']), dtype=torch.long),
                )

                # Mutant protein data
                mutant_data = Data(
                    x=torch.tensor(item['mutant_graph']['features'], dtype=torch.float),
                    edge_index=torch.tensor(item['mutant_graph']['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(item['mutant_graph']['edge_attr'], dtype=torch.float),
                    pos=torch.tensor(item['mutant_graph']['coords'], dtype=torch.float),
                    batch=torch.zeros(len(item['mutant_graph']['features']), dtype=torch.long),
                )

                # RNA data
                rna_data = Data(
                    x=torch.tensor(item['rna_graph']['features'], dtype=torch.float),
                    edge_index=torch.tensor(item['rna_graph']['edge_index'], dtype=torch.long),
                    edge_attr=torch.tensor(item['rna_graph']['edge_attr'], dtype=torch.float),
                    pos=torch.tensor(item['rna_graph']['coords'], dtype=torch.float),
                    batch=torch.zeros(len(item['rna_graph']['features']), dtype=torch.long),
                )

                # 生成对接口索引
                if 'interface' in item['wild_graph'] and item['wild_graph']['interface'] is not None:
                    wild_data.interface_index = self._get_interface_index(item['wild_graph']['interface'])
                    wild_data.interface_features = self._extract_interface_features(item['wild_graph']['interface'])
                else:
                    wild_data.interface_index = torch.zeros((2, 0), dtype=torch.long)
                    wild_data.interface_features = None

                if 'interface' in item['mutant_graph'] and item['mutant_graph']['interface'] is not None:
                    mutant_data.interface_index = self._get_interface_index(item['mutant_graph']['interface'])
                    mutant_data.interface_features = self._extract_interface_features(item['mutant_graph']['interface'])
                else:
                    mutant_data.interface_index = torch.zeros((2, 0), dtype=torch.long)
                    mutant_data.interface_features = None

                # 生成方向矩阵
                wild_data.orient = self._generate_orientation_matrices(wild_data.pos)
                mutant_data.orient = self._generate_orientation_matrices(mutant_data.pos)
                rna_data.orient = self._generate_orientation_matrices(rna_data.pos)

                # 添加元数据
                wild_data.metadata = {
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'mutation': mutation,
                    'mutation_pos': mutation_pos
                }
                mutant_data.metadata = wild_data.metadata.copy()

                # 添加接触特征维度信息（如果有）
                if 'contact_feature_dim' in metadata:
                    wild_data.contact_feature_dim = metadata['contact_feature_dim']
                    mutant_data.contact_feature_dim = metadata['contact_feature_dim']
                    rna_data.contact_feature_dim = metadata['contact_feature_dim']

                # 添加统一图信息（如果有）
                if 'unified_graph' in item and item['unified_graph'] is not None:
                    unified = item['unified_graph']
                    wild_data.unified_features = torch.tensor(unified['node_features'], dtype=torch.float)
                    wild_data.unified_edge_index = torch.tensor(unified['edge_index'], dtype=torch.long)
                    wild_data.unified_edge_attr = torch.tensor(unified['edge_attr'], dtype=torch.float)
                    wild_data.unified_pos = torch.tensor(unified['pos'], dtype=torch.float)

                    # 添加掩码
                    wild_data.wild_mask = torch.tensor(unified['wild_mask'], dtype=torch.bool)
                    wild_data.mutant_mask = torch.tensor(unified['mutant_mask'], dtype=torch.bool)
                    wild_data.rna_mask = torch.tensor(unified['rna_mask'], dtype=torch.bool)
                    wild_data.interface_mask = torch.tensor(unified['interface_mask'], dtype=torch.bool)

                    # 共享统一图信息
                    mutant_data.unified_features = wild_data.unified_features
                    mutant_data.unified_edge_index = wild_data.unified_edge_index
                    mutant_data.unified_edge_attr = wild_data.unified_edge_attr
                    mutant_data.unified_pos = wild_data.unified_pos
                    mutant_data.wild_mask = wild_data.wild_mask
                    mutant_data.mutant_mask = wild_data.mutant_mask
                    mutant_data.rna_mask = wild_data.rna_mask
                    mutant_data.interface_mask = wild_data.interface_mask

                # === 记录原始特征维度 ===
                original_wild_dim = wild_data.x.size(1)
                original_mutant_dim = mutant_data.x.size(1)
                original_rna_dim = rna_data.x.size(1)

                # ===== ESM2特征处理开始 =====
                if self.use_esm2_features:
                    # 加载野生型ESM2特征
                    wild_esm2_features = self.load_esm2_features(pdb_id, chain_id, mutation, is_wild=True)
                    mutant_esm2_features = self.load_esm2_features(pdb_id, chain_id, mutation, is_wild=False)

                    if wild_esm2_features is not None and mutant_esm2_features is not None:
                        # 检查ESM2特征维度是否与原始节点数匹配
                        wild_seq_len = wild_data.x.size(0)
                        mutant_seq_len = mutant_data.x.size(0)

                        if (wild_esm2_features.size(0) == wild_seq_len and
                                mutant_esm2_features.size(0) == mutant_seq_len):

                            # 根据feature_type处理ESM2特征
                            if self.feature_type == 4:
                                # 纯ESM2特征，直接替换原始蛋白质特征
                                wild_data.x = wild_esm2_features
                                mutant_data.x = mutant_esm2_features

                            elif self.feature_type in [5, 6, 7]:
                                # ESM2 + 多尺度特征，先保存ESM2特征，稍后合并
                                wild_data.esm2_features = wild_esm2_features
                                mutant_data.esm2_features = mutant_esm2_features

                            esm2_loaded_count += 1

                        else:
                            # 维度不匹配
                            esm2_dimension_mismatch += 1
                            if self.check_esm2_features:
                                print(f"维度不匹配 {pdb_id}_{chain_id}_{mutation}: "
                                      f"野生型 ESM2({wild_esm2_features.size(0)}) vs 节点数({wild_seq_len}), "
                                      f"突变型 ESM2({mutant_esm2_features.size(0)}) vs 节点数({mutant_seq_len})")

                            # 如果是纯ESM2特征类型但维度不匹配，跳过此样本
                            if self.feature_type == 4:
                                print(f"跳过样本 {i}: ESM2特征维度不匹配且feature_type=4")
                                continue
                    else:
                        esm2_missing_count += 1
                        if self.check_esm2_features:
                            print(f"ESM2特征缺失: {pdb_id}_{chain_id}_{mutation}")

                        # 如果是纯ESM2特征类型但特征缺失，跳过此样本
                        if self.feature_type == 4:
                            print(f"跳过样本 {i}: ESM2特征缺失且feature_type=4")
                            continue
                # ===== ESM2特征处理结束 =====

                # === 计算节点级接触特征（新增功能）===
                if self.compute_node_features:
                    # 为节点级特征创建缓存路径
                    wild_node_cache_path = self.get_node_contact_cache_path(pdb_id, chain_id, mutation, is_wild=True)
                    mutant_node_cache_path = self.get_node_contact_cache_path(pdb_id, chain_id, mutation, is_wild=False)

                    # 计算野生型节点级特征
                    should_compute_wild_node = True

                    if self.use_cached_contacts and not self.force_recompute and wild_node_cache_path and os.path.exists(wild_node_cache_path):
                        try:
                            cached_data = torch.load(wild_node_cache_path)
                            cached_thresholds = cached_data.get('thresholds')

                            if self._check_thresholds_match(cached_thresholds, self.contact_thresholds):
                                wild_node_contact_dist = cached_data['node_contact_dist']
                                wild_node_contact_int = cached_data['node_contact_int']

                                # 添加维度检查
                                if wild_node_contact_dist.size(0) != len(wild_data.pos):
                                    print(
                                        f"缓存的野生型节点数 ({wild_node_contact_dist.size(0)}) 与当前节点数 ({len(wild_data.pos)}) 不匹配，重新计算")
                                    should_compute_wild_node = True
                                else:
                                    cached_node_count += 1
                                    should_compute_wild_node = False
                                    wild_data.node_contact_dist = wild_node_contact_dist
                                    wild_data.node_contact_int = wild_node_contact_int
                        except Exception as e:
                            print(f"读取节点特征缓存时出错: {str(e)}")
                            should_compute_wild_node = True

                    if should_compute_wild_node:
                        wild_coords = wild_data.pos
                        rna_coords = rna_data.pos

                        wild_node_contact_dist, wild_node_contact_int = self.compute_node_contacts(
                            wild_coords, rna_coords, self.contact_thresholds
                        )

                        # 保存到数据对象但不合并到特征
                        wild_data.node_contact_dist = wild_node_contact_dist
                        wild_data.node_contact_int = wild_node_contact_int

                        # 缓存结果
                        if wild_node_cache_path:
                            torch.save({
                                'node_contact_dist': wild_node_contact_dist,
                                'node_contact_int': wild_node_contact_int,
                                'thresholds': self.contact_thresholds
                            }, wild_node_cache_path)
                        computed_node_count += 1

                    # 计算突变型节点级特征
                    should_compute_mutant_node = True

                    if self.use_cached_contacts and not self.force_recompute and mutant_node_cache_path and os.path.exists(mutant_node_cache_path):
                        try:
                            cached_data = torch.load(mutant_node_cache_path)
                            cached_thresholds = cached_data.get('thresholds')

                            if self._check_thresholds_match(cached_thresholds, self.contact_thresholds):
                                mutant_node_contact_dist = cached_data['node_contact_dist']
                                mutant_node_contact_int = cached_data['node_contact_int']
                                cached_node_count += 1
                                should_compute_mutant_node = False

                                # 保存到数据对象但不合并到特征
                                mutant_data.node_contact_dist = mutant_node_contact_dist
                                mutant_data.node_contact_int = mutant_node_contact_int
                        except Exception as e:
                            print(f"读取节点特征缓存时出错: {str(e)}")

                    if should_compute_mutant_node:
                        mutant_coords = mutant_data.pos
                        rna_coords = rna_data.pos

                        mutant_node_contact_dist, mutant_node_contact_int = self.compute_node_contacts(
                            mutant_coords, rna_coords, self.contact_thresholds
                        )

                        # 保存到数据对象但不合并到特征
                        mutant_data.node_contact_dist = mutant_node_contact_dist
                        mutant_data.node_contact_int = mutant_node_contact_int

                        # 缓存结果
                        if mutant_node_cache_path:
                            torch.save({
                                'node_contact_dist': mutant_node_contact_dist,
                                'node_contact_int': mutant_node_contact_int,
                                'thresholds': self.contact_thresholds
                            }, mutant_node_cache_path)
                        computed_node_count += 1

                    # 计算RNA节点的接触特征
                    # 使用明确的命名避免与之前的变量冲突
                    rna_pos_for_contacts = rna_data.pos.clone()  # 克隆以避免修改原始数据
                    wild_pos_for_contacts = wild_data.pos.clone()

                    # 确保坐标在CPU上
                    if rna_pos_for_contacts.is_cuda:
                        rna_pos_for_contacts = rna_pos_for_contacts.cpu()
                    if wild_pos_for_contacts.is_cuda:
                        wild_pos_for_contacts = wild_pos_for_contacts.cpu()

                    # 添加调试信息
                    # print(f"RNA节点计算: 特征节点数={rna_data.x.size(0)}, 坐标节点数={rna_pos_for_contacts.size(0)}")

                    # 计算RNA接触特征 - 使用新的变量名
                    rna_wild_contact_dist, rna_wild_contact_int = self.compute_node_contacts(
                        rna_pos_for_contacts, wild_pos_for_contacts, self.contact_thresholds
                    )

                    # 直接使用计算结果
                    rna_contact_dist = rna_wild_contact_dist
                    rna_contact_int = rna_wild_contact_int

                    # 保存到数据对象
                    rna_data.node_contact_dist = rna_contact_dist
                    rna_data.node_contact_int = rna_contact_int

                    # # 注意：这里我们需要反向计算RNA与蛋白质之间的接触
                    # rna_wild_contact_dist, rna_wild_contact_int = self.compute_node_contacts(
                    #     rna_coords, wild_coords, self.contact_thresholds
                    # )

                    rna_contact_dist = rna_wild_contact_dist
                    rna_contact_int = rna_wild_contact_int
                    # rna_mutant_contact_dist, rna_mutant_contact_int = self.compute_node_contacts(
                    #     rna_coords, mutant_coords, self.contact_thresholds
                    # )
                    #
                    # # 取平均作为RNA的接触特征
                    # rna_contact_dist = (rna_wild_contact_dist + rna_mutant_contact_dist) / 2.0
                    # rna_contact_int = (rna_wild_contact_int + rna_mutant_contact_int) / 2.0

                    # 保存到数据对象但不合并
                    rna_data.node_contact_dist = rna_contact_dist
                    rna_data.node_contact_int = rna_contact_int

                    # 如果需要合并特征
                    if self.merge_features:
                        # 根据feature_type确定要合并的特征
                        if self.feature_type == 1:  # 仅使用分布特征
                            # 合并野生型蛋白质特征
                            wild_data.x = torch.cat([
                                wild_data.x,  # 原始特征
                                wild_data.node_contact_dist  # 接触分布特征
                            ], dim=1)

                            # 合并突变型蛋白质特征
                            mutant_data.x = torch.cat([
                                mutant_data.x,  # 原始特征
                                mutant_data.node_contact_dist  # 接触分布特征
                            ], dim=1)

                            # 合并RNA特征
                            rna_data.x = torch.cat([
                                rna_data.x,  # 原始特征
                                rna_data.node_contact_dist  # 接触分布特征
                            ], dim=1)

                        elif self.feature_type == 2:  # 仅使用强度特征
                            # 合并野生型蛋白质特征
                            wild_data.x = torch.cat([
                                wild_data.x,  # 原始特征
                                wild_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                            # 合并突变型蛋白质特征
                            mutant_data.x = torch.cat([
                                mutant_data.x,  # 原始特征
                                mutant_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                            # 合并RNA特征
                            rna_data.x = torch.cat([
                                rna_data.x,  # 原始特征
                                rna_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                        elif self.feature_type == 3:  # 完整多尺度特征
                            # 合并野生型蛋白质特征
                            wild_data.x = torch.cat([
                                wild_data.x,  # 原始特征
                                wild_data.node_contact_dist,  # 接触分布特征
                                wild_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                            # 合并突变型蛋白质特征
                            mutant_data.x = torch.cat([
                                mutant_data.x,  # 原始特征
                                mutant_data.node_contact_dist,  # 接触分布特征
                                mutant_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                            # 合并RNA特征
                            rna_data.x = torch.cat([
                                rna_data.x,  # 原始特征
                                rna_data.node_contact_dist,  # 接触分布特征
                                rna_data.node_contact_int  # 接触强度特征
                            ], dim=1)

                        # 记录新的特征维度
                        wild_data.feature_dims = {
                            'original': original_wild_dim,
                            'contact_dist': wild_data.node_contact_dist.size(1),
                            'contact_int': wild_data.node_contact_int.size(1),
                            'total': wild_data.x.size(1)
                        }

                        mutant_data.feature_dims = wild_data.feature_dims.copy()

                        rna_data.feature_dims = {
                            'original': original_rna_dim,
                            'contact_dist': rna_data.node_contact_dist.size(1),
                            'contact_int': rna_data.node_contact_int.size(1),
                            'total': rna_data.x.size(1)
                        }

                # === 计算块级接触特征（与原代码类似）===
                if self.compute_block_features:
                    # 检查缓存
                    wild_block_cache_path = self.get_block_contact_cache_path(pdb_id, chain_id, mutation, is_wild=True)
                    mutant_block_cache_path = self.get_block_contact_cache_path(pdb_id, chain_id, mutation, is_wild=False)

                    # 计算野生型块级接触
                    should_compute_wild_block = True

                    if self.use_cached_contacts and not self.force_recompute and wild_block_cache_path and os.path.exists(wild_block_cache_path):
                        try:
                            cached_data = torch.load(wild_block_cache_path)
                            cached_thresholds = cached_data.get('thresholds')

                            if self._check_thresholds_match(cached_thresholds, self.contact_thresholds):
                                wild_data.block_contact_dist = cached_data['contact_dist']
                                wild_data.block_contact_int = cached_data['contact_int']
                                wild_data.chunk_size = cached_data['chunk_size']
                                wild_data.contact_thresholds = cached_data['thresholds']
                                cached_block_count += 1
                                should_compute_wild_block = False
                        except Exception as e:
                            print(f"读取块级特征缓存时出错: {str(e)}")

                    if should_compute_wild_block:
                        wild_coords = wild_data.pos
                        rna_coords = rna_data.pos

                        wild_block_dist, wild_block_int = self.compute_block_contacts(
                            wild_coords, rna_coords, self.chunk_size, self.contact_thresholds
                        )

                        # 保存到数据对象
                        wild_data.block_contact_dist = wild_block_dist
                        wild_data.block_contact_int = wild_block_int
                        wild_data.chunk_size = self.chunk_size
                        wild_data.contact_thresholds = self.contact_thresholds

                        # 缓存结果
                        if wild_block_cache_path:
                            torch.save({
                                'contact_dist': wild_block_dist,
                                'contact_int': wild_block_int,
                                'chunk_size': self.chunk_size,
                                'thresholds': self.contact_thresholds
                            }, wild_block_cache_path)
                        computed_block_count += 1

                    # 计算突变型块级接触
                    should_compute_mutant_block = True

                    if self.use_cached_contacts and not self.force_recompute and mutant_block_cache_path and os.path.exists(mutant_block_cache_path):
                        try:
                            cached_data = torch.load(mutant_block_cache_path)
                            cached_thresholds = cached_data.get('thresholds')

                            if self._check_thresholds_match(cached_thresholds, self.contact_thresholds):
                                mutant_data.block_contact_dist = cached_data['contact_dist']
                                mutant_data.block_contact_int = cached_data['contact_int']
                                mutant_data.chunk_size = cached_data['chunk_size']
                                mutant_data.contact_thresholds = cached_data['thresholds']
                                cached_block_count += 1
                                should_compute_mutant_block = False
                        except Exception as e:
                            print(f"读取块级特征缓存时出错: {str(e)}")

                    if should_compute_mutant_block:
                        mutant_coords = mutant_data.pos
                        rna_coords = rna_data.pos

                        mutant_block_dist, mutant_block_int = self.compute_block_contacts(
                            mutant_coords, rna_coords, self.chunk_size, self.contact_thresholds
                        )

                        # 保存到数据对象
                        mutant_data.block_contact_dist = mutant_block_dist
                        mutant_data.block_contact_int = mutant_block_int
                        mutant_data.chunk_size = self.chunk_size
                        mutant_data.contact_thresholds = self.contact_thresholds

                        # 缓存结果
                        if mutant_block_cache_path:
                            torch.save({
                                'contact_dist': mutant_block_dist,
                                'contact_int': mutant_block_int,
                                'chunk_size': self.chunk_size,
                                'thresholds': self.contact_thresholds
                            }, mutant_block_cache_path)
                        computed_block_count += 1

                # ===== 特征合并逻辑（针对ESM2 + 多尺度特征）=====
                if (self.use_esm2_features and self.feature_type in [5, 6, 7] and
                        hasattr(wild_data, 'esm2_features')):

                    # 确保有接触特征
                    if hasattr(wild_data, 'node_contact_dist') and hasattr(wild_data, 'node_contact_int'):
                        # 根据feature_type确定要合并的特征
                        if self.feature_type == 5:  # ESM2+分布特征
                            wild_data.x = torch.cat([
                                wild_data.esm2_features,
                                wild_data.node_contact_dist
                            ], dim=1)
                            mutant_data.x = torch.cat([
                                mutant_data.esm2_features,
                                mutant_data.node_contact_dist
                            ], dim=1)

                            # 为RNA也添加接触分布特征（如果计算了的话）
                            if hasattr(rna_data, 'node_contact_dist'):
                                rna_data.x = torch.cat([
                                    rna_data.x,  # 原始RNA特征
                                    rna_data.node_contact_dist
                                ], dim=1)

                        elif self.feature_type == 6:  # ESM2+强度特征
                            wild_data.x = torch.cat([
                                wild_data.esm2_features,
                                wild_data.node_contact_int
                            ], dim=1)
                            mutant_data.x = torch.cat([
                                mutant_data.esm2_features,
                                mutant_data.node_contact_int
                            ], dim=1)

                            # 为RNA也添加接触强度特征
                            if hasattr(rna_data, 'node_contact_int'):
                                rna_data.x = torch.cat([
                                    rna_data.x,  # 原始RNA特征
                                    rna_data.node_contact_int
                                ], dim=1)

                        elif self.feature_type == 7:  # ESM2+完整多尺度特征
                            wild_data.x = torch.cat([
                                wild_data.esm2_features,
                                wild_data.node_contact_dist,
                                wild_data.node_contact_int
                            ], dim=1)
                            mutant_data.x = torch.cat([
                                mutant_data.esm2_features,
                                mutant_data.node_contact_dist,
                                mutant_data.node_contact_int
                            ], dim=1)

                            # 为RNA也添加完整接触特征
                            if (hasattr(rna_data, 'node_contact_dist') and
                                    hasattr(rna_data, 'node_contact_int')):
                                rna_data.x = torch.cat([
                                    rna_data.x,  # 原始RNA特征
                                    rna_data.node_contact_dist,
                                    rna_data.node_contact_int
                                ], dim=1)

                    else:
                        # 如果没有接触特征，只使用ESM2特征
                        wild_data.x = wild_data.esm2_features
                        mutant_data.x = mutant_data.esm2_features
                # ===== 特征合并逻辑结束 =====

                # 将样本添加到列表
                pyg_data_list.append((wild_data, mutant_data, rna_data, ddg))

                # 原有代码：每100个样本打印一次进度（🔥 更新统计信息）
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(dataset)} samples")
                    if self.compute_node_features:
                        print(f"Node features: {cached_node_count} cached, {computed_node_count} computed")
                    if self.compute_block_features:
                        print(f"Block features: {cached_block_count} cached, {computed_block_count} computed")
                    # 🔥 新增：ESM2统计信息
                    if self.use_esm2_features:
                        print(
                            f"ESM2特征: {esm2_loaded_count} 成功加载, {esm2_missing_count} 缺失, {esm2_dimension_mismatch} 维度不匹配")
                    # 🔥 新增：显示当前特征维度
                    if self.merge_features and self.compute_node_features:
                        print(f"合并后特征维度 - 蛋白质: {wild_data.x.size(1)}, RNA: {rna_data.x.size(1)}")

                    # 🔥 显示合并后的特征维度
                    if self.merge_features and len(pyg_data_list) > 0:
                        sample_wild = pyg_data_list[-1][0]  # 最新处理的样本
                        sample_rna = pyg_data_list[-1][2]
                        print(f"当前特征维度 - 蛋白质: {sample_wild.x.size(1)}, RNA: {sample_rna.x.size(1)}")


            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        print(f"Successfully converted {len(pyg_data_list)} samples out of {len(dataset)}")
        if self.compute_node_features:
            print(f"Node contact features: {cached_node_count} loaded from cache, {computed_node_count} newly computed")
        if self.compute_block_features:
            print(
                f"Block contact features: {cached_block_count} loaded from cache, {computed_block_count} newly computed")

        # 新增：ESM2特征最终统计
        if self.use_esm2_features:
            print(f"ESM2特征最终统计:")
            print(f"  成功加载: {esm2_loaded_count}")
            print(f"  特征缺失: {esm2_missing_count}")
            print(f"  维度不匹配: {esm2_dimension_mismatch}")

        # 打印特征维度
        if len(pyg_data_list) > 0 and self.merge_features and self.compute_node_features:
            wild_data = pyg_data_list[0][0]
            rna_data = pyg_data_list[0][2]
            print(f"Final feature dimensions:")
            print(f"  Protein: {wild_data.x.size(1)} (Original: {original_wild_dim} + "
                  f"Contact Dist: {wild_data.node_contact_dist.size(1)} + "
                  f"Contact Int: {wild_data.node_contact_int.size(1)})")
            print(f"  RNA: {rna_data.x.size(1)} (Original: {original_rna_dim} + "
                  f"Contact Dist: {rna_data.node_contact_dist.size(1)} + "
                  f"Contact Int: {rna_data.node_contact_int.size(1)})")

            # 🔥 显示特征组成详情
            if self.use_esm2_features:
                if self.feature_type == 4:
                    print(f"  蛋白质特征组成: 纯ESM2 (1280维)")
                elif self.feature_type == 5:
                    print(f"  蛋白质特征组成: ESM2 (1280维) + 分布特征 ({wild_data.x.size(1) - 1280}维)")
                elif self.feature_type == 6:
                    print(f"  蛋白质特征组成: ESM2 (1280维) + 强度特征 ({wild_data.x.size(1) - 1280}维)")
                elif self.feature_type == 7:
                    print(f"  蛋白质特征组成: ESM2 (1280维) + 完整多尺度特征 ({wild_data.x.size(1) - 1280}维)")


        return pyg_data_list

    def _get_interface_index(self, interface_data):
        """提取界面边索引"""
        if not interface_data or not interface_data.get('pairs', []):
            return torch.zeros((2, 0), dtype=torch.long)

        protein_indices = []
        rna_indices = []

        for pair in interface_data['pairs']:
            protein_indices.append(pair['protein_idx'])
            rna_indices.append(pair['rna_idx'])

        interface_index = torch.tensor([protein_indices, rna_indices], dtype=torch.long)
        return interface_index

    def _extract_interface_features(self, interface_data):
        """提取详细的界面特征"""
        if not interface_data or not interface_data.get('pairs', []):
            return None

        pairs = interface_data['pairs']

        protein_indices = torch.tensor([p['protein_idx'] for p in pairs], dtype=torch.long)
        rna_indices = torch.tensor([p['rna_idx'] for p in pairs], dtype=torch.long)
        distances = torch.tensor([p['distance'] for p in pairs], dtype=torch.float)
        thresholds = torch.tensor([p['threshold'] for p in pairs], dtype=torch.float)

        return {
            'protein_indices': protein_indices,
            'rna_indices': rna_indices,
            'distances': distances,
            'thresholds': thresholds
        }

    def _generate_orientation_matrices(self, positions):
        """生成方向矩阵（局部坐标系）"""
        n_nodes = positions.size(0)

        orientations = torch.eye(3, dtype=torch.float).unsqueeze(0).repeat(n_nodes, 1, 1)

        if n_nodes > 1:
            for i in range(1, n_nodes):
                angle = (i % 10) * 0.1
                cos_val = torch.cos(torch.tensor(angle))
                sin_val = torch.sin(torch.tensor(angle))

                rot_matrix = torch.eye(3, dtype=torch.float)
                rot_matrix[0, 0] = cos_val
                rot_matrix[0, 1] = -sin_val
                rot_matrix[1, 0] = sin_val
                rot_matrix[1, 1] = cos_val

                orientations[i] = rot_matrix

        return orientations

    def add_reverse_mutations(self):
        """添加反向突变样本（突变型作为野生型，野生型作为突变型，DDG取负）"""
        # 创建新的数据列表，交错排列正向和反向突变
        new_data_list = []

        for wild_data, mutant_data, rna_data, ddg in self.data_list:
            # 添加原始正向突变
            new_data_list.append((wild_data, mutant_data, rna_data, ddg))

            # 创建并添加对应的反向突变
            reverse_sample = (mutant_data, wild_data, rna_data, -ddg)
            new_data_list.append(reverse_sample)

        # 用新的交错列表替换原始数据列表
        self.data_list = new_data_list

    def split_dataset(self, val_ratio=0.15, test_ratio=0.15, strategy=None, train_ratio=None):
        """
        Custom dataset splitting method supporting various strategies.

        Parameters:
            strategy: Splitting strategy to use
                - None or 'random': Default random splitting
                - 'pdb_limited': Limit training samples per PDB
                - 'train_ratio': Split based on specified training ratio
            val_ratio: Validation set ratio (for random strategy)
            test_ratio: Test set ratio (for random strategy)
            train_ratio: Training set ratio (only used with 'train_ratio' strategy)

        Returns:
            train_data, val_data, test_data: Dataset splits
        """
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Use default splitting if no strategy specified
        if strategy is None or strategy == 'random':
            print(f"Using default random splitting strategy with val_ratio={val_ratio}, test_ratio={test_ratio}")
            return self._random_split(val_ratio, test_ratio)

        # PDB-limited training split
        elif strategy == 'pdb_limited':
            print(f"Using PDB-limited splitting with samples per PDB")
            return self._pdb_limited_training_split(val_ratio, test_ratio)

        # Training ratio split
        elif strategy == 'train_ratio':
            if train_ratio is None:
                raise ValueError("train_ratio must be specified when using 'train_ratio' strategy")
            print(f"Using training ratio splitting with train_ratio={train_ratio}")
            return self._train_ratio_split(train_ratio)

        # Fallback to default splitting
        else:
            print(f"Unknown strategy '{strategy}', falling back to default splitting")
            return self._random_split(val_ratio, test_ratio)

    def _train_ratio_split(self, train_ratio):
        """
        Split dataset based on specified training ratio.

        Parameters:
            train_ratio: Fraction of data to use for training (0.1-0.9)

        Returns:
            train_data, val_data, test_data: Dataset splits
        """
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Sample total count
        total_samples = len(self.data_list)

        # Calculate train size
        train_size = int(total_samples * train_ratio)

        # Remaining data will be split 50/50 between validation and test
        remaining = total_samples - train_size
        val_size = remaining // 2
        test_size = remaining - val_size

        # Shuffle indices
        indices = list(range(total_samples))
        if self.shuffle:
            random.shuffle(indices)

        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create datasets
        train_data = [self.data_list[i] for i in train_indices]
        val_data = [self.data_list[i] for i in val_indices]
        test_data = [self.data_list[i] for i in test_indices]

        print(f"Training ratio split:")
        print(f"  Training: {len(train_data)} samples ({train_ratio:.1%})")
        print(f"  Validation: {len(val_data)} samples ({val_size / total_samples:.1%})")
        print(f"  Test: {len(test_data)} samples ({test_size / total_samples:.1%})")

        return train_data, val_data, test_data

    def _pdb_limited_training_split(self, val_ratio=0.15, test_ratio=0.15):
        """
        Two-stage splitting strategy:
        1. First limit training samples to 2 per PDB ID
        2. Then distribute remaining samples into validation and test sets

        This tests generalization while maintaining enough training signal
        """
        # Number of samples per PDB to use for training
        train_samples_per_pdb = 2  # 2

        # Group samples by PDB ID
        pdb_groups = {}
        for idx, (wild_data, mutant_data, rna_data, ddg) in enumerate(self.data_list):
            pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
            if pdb_id not in pdb_groups:
                pdb_groups[pdb_id] = []
            pdb_groups[pdb_id].append(idx)

        # Select limited samples for training from each PDB
        train_indices = []
        remaining_indices = []

        for pdb_id, indices in pdb_groups.items():
            if len(indices) <= train_samples_per_pdb:
                # If very few samples, use all for training
                train_indices.extend(indices)
            else:
                # Randomly select N samples per PDB for training
                selected = random.sample(indices, train_samples_per_pdb)
                train_indices.extend(selected)
                remaining_indices.extend([idx for idx in indices if idx not in selected])

        # Calculate sizes for validation and test
        remaining_total = len(remaining_indices)
        # Adjust ratios to apply to remaining samples
        adjusted_val_ratio = val_ratio / (val_ratio + test_ratio)
        val_size = int(remaining_total * adjusted_val_ratio)

        # Split remaining indices into validation and test
        random.shuffle(remaining_indices)
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]

        # Create data splits
        train_data = [self.data_list[i] for i in train_indices]
        val_data = [self.data_list[i] for i in val_indices]
        test_data = [self.data_list[i] for i in test_indices]

        # Print split summary
        print(f"PDB-limited split summary:")
        print(f"  Training: {len(train_data)} samples from {len(pdb_groups)} PDBs")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")

        # Count sample distribution in training set
        train_pdb_counts = {}
        for idx in train_indices:
            wild_data = self.data_list[idx][0]
            pdb_id = wild_data.metadata.get('pdb_id', 'unknown')
            train_pdb_counts[pdb_id] = train_pdb_counts.get(pdb_id, 0) + 1

        # Sample distribution statistics
        counts = list(train_pdb_counts.values())
        avg_samples = sum(counts) / len(counts) if counts else 0
        max_samples = max(counts) if counts else 0
        print(
            f"  Training set PDB sample distribution: Average {avg_samples:.2f} samples/PDB, Max {max_samples} samples/PDB")

        return train_data, val_data, test_data

    def _random_split(self, val_ratio, test_ratio):
        """拆分为训练集、验证集和测试集"""
        # 设置随机种子
        random.seed(self.seed)

        # 样本总数
        total_samples = len(self.data_list)

        # 计算各集合大小
        test_size = int(total_samples * test_ratio)
        val_size = int(total_samples * val_ratio)
        train_size = total_samples - test_size - val_size

        # 打乱索引
        indices = list(range(total_samples))
        if self.shuffle:
            random.shuffle(indices)

        # 拆分索引
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size + val_size]
        train_indices = indices[test_size + val_size:]

        # 创建数据集
        train_data = [self.data_list[i] for i in train_indices]
        val_data = [self.data_list[i] for i in val_indices]
        test_data = [self.data_list[i] for i in test_indices]

        print(f"Dataset split complete:")
        print(f"  Training: {len(train_data)} samples ({len(train_indices) / total_samples:.1%})")
        print(f"  Validation: {len(val_data)} samples ({len(val_indices) / total_samples:.1%})")
        print(f"  Test: {len(test_data)} samples ({len(test_indices) / total_samples:.1%})")

        return train_data, val_data, test_data

    def create_data_loaders(self):
        """创建数据加载器"""
        # 使用自定义的收集函数处理三元组结构
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_protein_rna_triple,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )

        val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_protein_rna_triple,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_protein_rna_triple,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )

        return train_loader, val_loader, test_loader

    def get_esm2_cache_path(self, pdb_id, chain_id, mutation, is_wild=True):
        """获取ESM2特征缓存文件路径"""
        if not self.esm2_cache_dir:
            return None

        protein_type = "wild" if is_wild else "mutant"
        filename = f"{pdb_id}_{chain_id}_{protein_type}_{mutation}_esm2.pt"
        return os.path.join(self.esm2_cache_dir, filename)

    def load_esm2_features(self, pdb_id, chain_id, mutation, is_wild=True):
        """
        加载ESM2特征

        Returns:
            torch.Tensor: ESM2特征，形状为 [seq_len, 1280]
            None: 如果加载失败
        """
        cache_path = self.get_esm2_cache_path(pdb_id, chain_id, mutation, is_wild)

        if not cache_path or not os.path.exists(cache_path):
            if self.check_esm2_features:
                protein_type = "wild" if is_wild else "mutant"
                print(f"⚠️ ESM2特征文件不存在: {pdb_id}_{chain_id}_{protein_type}_{mutation}")
            return None

        try:
            esm2_data = torch.load(cache_path, map_location='cpu')

            # 提取特征数据
            if isinstance(esm2_data, dict):
                features = esm2_data.get('features')
                if features is None:
                    if self.check_esm2_features:
                        print(f"⚠️ ESM2文件中没有'features'字段: {cache_path}")
                    return None
            else:
                features = esm2_data

            # 确保是torch.Tensor
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            return features

        except Exception as e:
            if self.check_esm2_features:
                print(f"❌ 加载ESM2特征失败 {cache_path}: {str(e)}")
            return None

def collate_protein_rna_triple(batch):
    """
    自定义收集函数，处理(wild, mutant, rna, ddg)元组
    """
    # 分离组件
    wild_list = [item[0] for item in batch]
    mutant_list = [item[1] for item in batch]
    rna_list = [item[2] for item in batch]
    ddg_list = [item[3] for item in batch]

    # 批处理每个组件
    wild_batch = Batch.from_data_list(wild_list)
    mutant_batch = Batch.from_data_list(mutant_list)
    rna_batch = Batch.from_data_list(rna_list)

    # 转换DDG值为张量
    if isinstance(ddg_list[0], torch.Tensor):
        ddg_tensor = torch.stack(ddg_list)
    else:
        ddg_tensor = torch.tensor(ddg_list, dtype=torch.float)

    return wild_batch, mutant_batch, rna_batch, ddg_tensor

def load_protein_rna_data(data_path, batch_size=1, val_ratio=0.15, test_ratio=0.15,
                          add_reverse=True, seed=42, shuffle=SHUFFLE, chunk_size=DEFAULT_CHUNK_SIZE,
                          contact_thresholds=DEFAULT_CONTACT_THRESHOLDS, cache_dir="./contact_cache",
                          use_cached_contacts=True, force_recompute=False, num_workers=NUM_WORKERS,
                          prefetch_factor=PREFETCH_FACTOR, compute_block_features=True, compute_node_features=True,
                          feature_type=3, split_strategy=None, train_ratio=None, esm2_cache_dir=DEFAULT_ESM2_CACHE_DIR, check_esm2_features=True):
    """
    Convenience function that returns enhanced protein-RNA interaction data loaders
    with node-level and block-level contact features.

    Parameters:
        data_path: Dataset path
        batch_size: Batch size
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        add_reverse: Whether to add reverse mutation samples
        seed: Random seed
        shuffle: Whether to shuffle data
        chunk_size: Block size
        contact_thresholds: Contact thresholds list
        cache_dir: Cache directory, None means no caching
        use_cached_contacts: Whether to use cached contact information
        force_recompute: Whether to force recompute contact information
        num_workers: Number of worker processes for DataLoader
        prefetch_factor: Prefetch factor for DataLoader
        compute_block_features: Whether to compute block-level contact features
        compute_node_features: Whether to compute node-level contact features
        feature_type: 多尺度特征类型 (0=无特征, 1=仅分布特征, 2=仅强度特征, 3=完整特征)
        split_strategy: Dataset splitting strategy (None, 'pdb_limited', 'directional', etc.)
        train_ratio: Training set ratio (only used with 'train_ratio' strategy)

    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    data_loader = EnhancedProteinRNADataLoader(
        data_path=data_path,
        batch_size=batch_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        add_reverse=add_reverse,
        seed=seed,
        shuffle=shuffle,
        chunk_size=chunk_size,
        contact_thresholds=contact_thresholds,
        cache_dir=cache_dir,
        use_cached_contacts=use_cached_contacts,
        force_recompute=force_recompute,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        compute_block_features=compute_block_features,
        compute_node_features=compute_node_features,
        feature_type=feature_type,
        split_strategy=split_strategy,
        train_ratio=train_ratio,  # Pass train_ratio parameter
        esm2_cache_dir=esm2_cache_dir,
        check_esm2_features=check_esm2_features
    )

    return data_loader.train_loader, data_loader.val_loader, data_loader.test_loader