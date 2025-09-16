"""
ProteinOnlyGraphTransformer - 专为蛋白质设计的Graph Transformer模型
- 只使用蛋白质数据，无需RNA数据
- 集成随机游走位置编码增强结构感知
- 严格遵循PyTorch Geometric的GPSConv官方实现
- 解决了边特征维度不匹配问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
from torch_geometric.nn import GINEConv, GPSConv, GATConv, GCNConv, GraphSAGE
from torch_geometric.transforms import AddRandomWalkPE


class EdgeProjector(nn.Module):
    """
    边特征投影模块，用于将边特征的维度调整为与节点特征匹配
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr):
        """投影边特征"""
        return self.projection(edge_attr)


class GraphTransformer(nn.Module):
    """
    专为蛋白质设计的Graph Transformer
    - 使用随机游走位置编码增强结构感知
    - 忽略RNA数据，只关注蛋白质突变效应
    - 严格基于PyG的官方GPSConv实现
    """
    def __init__(
            self,
            protein_channels,        # 蛋白质特征维度
            rna_channels=None,       # 为接口兼容性保留，但不使用
            hidden_channels=64,      # 隐藏层维度
            out_channels=1,          # 输出维度
            num_layers=3,            # GNN层数
            heads=4,                 # 注意力头数
            dropout=0.1,             # Dropout比例
            walk_length=16,          # 随机游走步数
            pe_dim=None,             # 位置编码维度，若None则使用hidden_channels//4
            use_performer=False,     # 是否使用Performer注意力
            edge_dim=16,             # 边特征维度，默认为16
            **kwargs                 # 其他参数
    ):
        super().__init__()

        # 保存配置
        self.protein_channels = protein_channels
        self.hidden_channels = hidden_channels
        self.walk_length = walk_length
        self.pe_dim = pe_dim if pe_dim is not None else hidden_channels // 4
        self.model_dim = hidden_channels
        self.edge_dim = edge_dim

        print(f"初始化蛋白质专用GraphTransformer - "
              f"特征维度: {protein_channels}, "
              f"隐藏维度: {hidden_channels}, "
              f"位置编码维度: {self.pe_dim}, "
              f"随机游走长度: {walk_length}, "
              f"边特征维度: {edge_dim}")

        # 随机游走位置编码
        self.pe_transform = AddRandomWalkPE(walk_length=walk_length, attr_name='pe')

        # 位置编码处理层
        self.pe_encoder = nn.Sequential(
            nn.Linear(walk_length, self.pe_dim),
            nn.LayerNorm(self.pe_dim),
            nn.GELU()
        )

        # 蛋白质特征编码器 - 输出维度减少以为位置编码留空间
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_channels, hidden_channels - self.pe_dim),
            nn.LayerNorm(hidden_channels - self.pe_dim),
            nn.GELU()
        )

        # 蛋白质类型嵌入
        self.protein_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels - self.pe_dim))

        # 边特征投影 - 将边特征投影到与节点特征匹配的维度
        self.edge_projector = EdgeProjector(edge_dim, hidden_channels)

        # Graph Transformer层堆叠
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            # 定义局部GNN
            nn_local = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )

            # 构建GINE卷积作为局部模型，明确指定edge_dim
            gine_conv = GINEConv(nn_local, edge_dim=hidden_channels)

            # 设置注意力类型和参数
            attn_type = 'performer' if use_performer else 'multihead'
            attn_kwargs = {'dropout': dropout}

            # 创建GPSConv层
            gps_conv = GPSConv(
                channels=hidden_channels,
                conv=gine_conv,
                heads=heads,
                dropout=dropout,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )

            self.gps_layers.append(gps_conv)

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def _add_position_encoding(self, data):
        """
        添加随机游走位置编码

        参数:
            data: 图数据

        返回:
            data: 添加了位置编码的图数据
        """
        # 如果数据中没有位置编码，应用随机游走变换
        if not hasattr(data, 'pe'):
            data = self.pe_transform(data)
        return data

    def encode_protein(self, data):
        """
        编码蛋白质数据

        参数:
            data: 蛋白质图数据

        返回:
            encoded: 编码后的特征
            batch: 批次信息
        """
        # 添加位置编码
        data = self._add_position_encoding(data)

        # 提取特征
        x, edge_index, pe = data.x, data.edge_index, data.pe
        batch = data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        # 编码节点特征
        protein_feats = self.protein_encoder(x)

        # 应用蛋白质类型嵌入
        protein_dense, mask = to_dense_batch(protein_feats, batch)
        mask = mask.bool()  # 确保掩码是bool类型
        protein_dense = protein_dense + self.protein_embedding
        protein_feats = protein_dense[mask]

        # 编码位置特征
        pe_feats = self.pe_encoder(pe)

        # 合并节点特征和位置编码
        combined_feats = torch.cat([protein_feats, pe_feats], dim=-1)

        # 投影边特征（如果存在）
        if edge_attr is not None:
            edge_attr = self.edge_projector(edge_attr)

        # 应用Graph Transformer层
        x = combined_feats
        for layer in self.gps_layers:
            try:
                x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
            except Exception as e:
                print(f"层处理出错: {str(e)}")
                # 这里可以添加更多的错误处理逻辑

        return x, batch

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        前向传播函数 - 忽略RNA数据，只使用蛋白质数据

        参数:
            wild_data: 野生型蛋白质图数据
            mutant_data: 突变型蛋白质图数据
            rna_data: RNA数据 (被忽略)

        返回:
            pred: 预测的ΔΔG值
        """
        # 编码野生型和突变型蛋白质
        wild_encoded, wild_batch = self.encode_protein(wild_data)
        mutant_encoded, mutant_batch = self.encode_protein(mutant_data)

        # 全局池化
        wild_pooled = scatter_mean(wild_encoded, wild_batch, dim=0)
        mutant_pooled = scatter_mean(mutant_encoded, mutant_batch, dim=0)

        # 计算差异特征
        diff_features = mutant_pooled - wild_pooled

        # 预测
        pred = self.predictor(diff_features).squeeze(-1)

        return pred


class GraphTransformer_WithRandomWalkPE(GraphTransformer):
    """
    蛋白质专用Graph Transformer - 包装类，用于模型工厂兼容
    """
    def __init__(
            self,
            protein_channels,
            rna_channels=None,
            hidden_channels=64,
            out_channels=1,
            num_layers=3,
            dropout=0.1,
            walk_length=16,
            edge_dim=16,  # 添加边特征维度参数
            **kwargs
    ):
        super().__init__(
            protein_channels=protein_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=max(1, hidden_channels // 16),
            dropout=dropout,
            walk_length=walk_length,
            edge_dim=edge_dim,  # 传递边特征维度
            **kwargs
        )
        print(f"初始化蛋白质专用GraphTransformer包装类 - 随机游走步长: {walk_length}, 边特征维度: {edge_dim}")

    @staticmethod
    def create_model(**kwargs):
        """
        静态工厂方法，用于模型工厂
        """
        return GraphTransformer_WithRandomWalkPE(**kwargs)


class GraphTransformer_AutoPE(GraphTransformer):
    """
    蛋白质专用Graph Transformer - 自动调整超参数版本
    """
    def __init__(
            self,
            protein_channels,
            rna_channels=None,
            hidden_channels=64,
            edge_dim=16,  # 添加边特征维度参数
            **kwargs
    ):
        # 根据输入特征维度和隐藏层大小自动调整超参数
        auto_params = {
            'out_channels': 1,
            'num_layers': min(5, max(2, hidden_channels // 32)),  # 根据隐藏层大小调整层数
            'heads': max(1, hidden_channels // 16),  # 根据隐藏层大小调整头数
            'dropout': 0.1,
            'walk_length': min(32, max(8, hidden_channels // 4)),  # 随机游走长度基于隐藏层大小
            'pe_dim': max(4, hidden_channels // 4),  # 位置编码维度
            'edge_dim': edge_dim  # 边特征维度
        }

        # 用户提供的参数覆盖自动参数
        for k, v in kwargs.items():
            if k in auto_params:
                auto_params[k] = v

        super().__init__(
            protein_channels=protein_channels,
            hidden_channels=hidden_channels,
            **auto_params
        )

        print(f"初始化自动调整版蛋白质GraphTransformer - 自动配置: {auto_params}")

    @staticmethod
    def create_model(**kwargs):
        """
        静态工厂方法，用于模型工厂
        """
        return GraphTransformer_AutoPE(**kwargs)