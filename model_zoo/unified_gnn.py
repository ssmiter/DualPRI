"""
统一的图神经网络模型实现 - 支持多种GNN类型的差分架构
可以通过参数选择GCN, GAT, GraphSAGE, GIN等不同类型的GNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GATv2Conv, GINConv, SAGEConv, PNAConv, EdgeConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.nn.models import MLP


class UnifiedGNN(nn.Module):
    """
    统一的图神经网络模型 - 使用差分架构预测蛋白质-RNA相互作用

    支持的GNN类型:
    - gcn: 图卷积网络
    - gat: 图注意力网络
    - gatv2: 图注意力网络v2
    - sage: GraphSAGE
    - gin: 图同构网络
    - pna: 主邻居聚合网络
    - edge: EdgeCNN
    """

    def __init__(
            self,
            protein_channels,  # 蛋白质特征维度
            rna_channels=None,  # RNA特征维度(为接口兼容)
            hidden_channels=64,  # 隐藏层维度
            out_channels=1,  # 输出维度
            num_layers=3,  # GNN层数
            dropout=0.1,  # Dropout比例
            gnn_type='gcn',  # GNN类型
            pool_type='mean',  # 池化类型: mean, sum, max
            norm=None,  # 归一化类型
            heads=4,  # GAT头数
            edge_dim=None,  # 边特征维度
            **kwargs
    ):
        super().__init__()

        # 保存配置
        self.protein_channels = protein_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.out_channels = out_channels

        # 如果未指定边维度，使用默认值16
        if edge_dim is None:
            self.edge_dim = 16  # 默认边特征维度
        else:
            self.edge_dim = edge_dim

        print(f"初始化统一GNN模型 - 类型: {gnn_type}, 蛋白质特征维度: {protein_channels}, "
              f"隐藏层维度: {hidden_channels}, 层数: {num_layers}, 头数: {heads}")

        # 蛋白质特征编码层
        self.protein_encoder = nn.Linear(protein_channels, hidden_channels)

        # 边特征编码层(如果GNN类型支持边特征)
        if self.gnn_type in ['gat', 'gatv2', 'pna', 'edge']:
            self.edge_encoder = nn.Linear(self.edge_dim, hidden_channels)

        # GNN层
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels if i > 0 else hidden_channels
            self.convs.append(self._create_conv_layer(in_ch, hidden_channels, gnn_type, heads))

        # 归一化层
        self.norms = None
        if norm == 'batch':
            self.norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
            ])
        elif norm == 'layer':
            self.norms = nn.ModuleList([
                nn.LayerNorm(hidden_channels) for _ in range(num_layers)
            ])

        # 全局池化函数
        if pool_type == 'mean':
            self.pool = global_mean_pool
        elif pool_type == 'sum':
            self.pool = global_add_pool
        elif pool_type == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"未知池化类型: {pool_type}")

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def _create_conv_layer(self, in_channels, out_channels, gnn_type, heads):
        """创建指定类型的GNN层"""
        if gnn_type == 'gcn':
            return GCNConv(in_channels, out_channels)
        elif gnn_type == 'gat':
            return GATConv(in_channels, out_channels // heads, heads=heads, concat=True)
        elif gnn_type == 'gatv2':
            return GATv2Conv(in_channels, out_channels // heads, heads=heads, concat=True)
        elif gnn_type == 'sage':
            return SAGEConv(in_channels, out_channels)
        elif gnn_type == 'gin':
            nn_layer = MLP([in_channels, out_channels, out_channels])
            return GINConv(nn_layer)
        elif gnn_type == 'pna':
            # 简化版PNA，实际应用中可能需要更多配置
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            return PNAConv(in_channels, out_channels,
                           aggregators=aggregators,
                           scalers=scalers,
                           deg=None)  # deg应该从数据中计算
        elif gnn_type == 'edge':
            mlp = MLP([2 * in_channels, out_channels, out_channels])
            return EdgeConv(mlp)
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        前向传播函数：计算野生型和突变型表示，然后取差值预测ΔΔG

        参数:
            wild_data: 野生型蛋白质图数据
            mutant_data: 突变型蛋白质图数据
            rna_data: RNA图数据 (可选，不使用)

        返回:
            pred: 预测的ΔΔG值
        """
        # 编码野生型蛋白质
        wild_x = self.encode_protein(wild_data)

        # 编码突变型蛋白质
        mutant_x = self.encode_protein(mutant_data)

        # 计算差异表示
        diff_x = mutant_x - wild_x

        # 预测
        pred = self.predictor(diff_x).squeeze(-1)

        return pred

    def encode_protein(self, data):
        """
        编码蛋白质数据

        参数:
            data: 蛋白质图数据

        返回:
            pooled_x: 编码后的蛋白质全局表示
        """
        x, edge_index = data.x, data.edge_index
        batch = data.batch

        # 获取边特征(如果存在且GNN类型支持)
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.gnn_type in ['gat', 'gatv2', 'pna',
                                                                                           'edge']:
            edge_attr = data.edge_attr
            if hasattr(self, 'edge_encoder'):
                edge_attr = self.edge_encoder(edge_attr)

        # 初始特征变换
        x = self.protein_encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 应用GNN层
        for i, conv in enumerate(self.convs):
            # 根据不同GNN类型调用不同的前向传播
            if self.gnn_type in ['gat', 'gatv2', 'pna']:
                if edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index)
            elif self.gnn_type == 'gin':
                x = conv(x, edge_index)
            elif self.gnn_type == 'edge':
                x = conv(x, edge_index)
            else:  # GCN, SAGE等
                x = conv(x, edge_index)

            # 应用归一化(如果有)
            if self.norms is not None:
                x = self.norms[i](x)

            # 激活和Dropout
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 全局池化
        pooled_x = self.pool(x, batch)

        return pooled_x

    @staticmethod
    def create_model(**kwargs):
        """
        静态工厂方法，用于模型工厂
        """
        # 从传入的参数中提取模型名称，用作gnn_type
        if 'model_name' in kwargs and 'gnn_type' not in kwargs:
            model_name = kwargs.pop('model_name').lower()
            if model_name in ['gcn', 'gat', 'gatv2', 'sage', 'gin', 'pna', 'edge']:
                kwargs['gnn_type'] = model_name

        return UnifiedGNN(**kwargs)