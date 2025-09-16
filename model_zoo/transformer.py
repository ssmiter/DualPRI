"""
Transformer模型 - 用于蛋白质-RNA相互作用预测
- 使用Performer注意力机制，线性复杂度O(n)
- 支持差分设计计算野生型和突变型之间的变化
- 双向处理增强特征提取能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean
from torch_geometric.nn.attention import PerformerAttention


class PerformerEncoderLayer(nn.Module):
    """
    使用Performer注意力的编码器层

    特点:
    - 线性复杂度的自注意力O(n)
    - 前馈网络和残差连接
    - 层归一化提高训练稳定性
    """

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()

        # Performer注意力
        self.self_attn = PerformerAttention(
            channels=d_model,
            heads=nhead,
            head_channels=d_model // nhead,
            dropout=dropout
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.gelu

    def forward(self, src, src_mask=None):
        """
        前向传播函数

        参数:
            src: [batch, seq_len, d_model] - 输入序列
            src_mask: 序列掩码 [batch, seq_len] - True表示有效位置

        返回:
            [batch, seq_len, d_model] - 处理后的序列
        """
        # 第一个子层：Performer注意力
        src2 = self.self_attn(src, mask=src_mask)
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 第二个子层：前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化

        return src


class BidirectionalPerformer(nn.Module):
    """
    双向Performer

    特点:
    - 前向与后向结合提取双向信息
    - 门控融合机制整合双向特征
    """

    def __init__(self, d_model, nhead=8, num_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 创建编码器层
        layers = nn.ModuleList([
            PerformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 前向和后向编码器
        self.forward_layers = nn.ModuleList(layers)
        self.backward_layers = nn.ModuleList([
            PerformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        """
        前向传播函数

        参数:
            x: [batch, seq_len, d_model] - 输入序列
            mask: [batch, seq_len] - 序列掩码 (True表示有效位置)

        返回:
            [batch, seq_len, d_model] - 处理后的序列
        """
        # 前向处理
        forward_out = x
        for layer in self.forward_layers:
            forward_out = layer(forward_out, mask)

        # 后向处理 - 翻转序列
        x_reverse = torch.flip(x, dims=[1])
        mask_reverse = torch.flip(mask, dims=[1])
        backward_out = x_reverse
        for layer in self.backward_layers:
            backward_out = layer(backward_out, mask_reverse)
        backward_out = torch.flip(backward_out, dims=[1])  # 翻转回来

        # 门控融合
        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate_values = self.gate(combined)
        fused = gate_values * forward_out + (1 - gate_values) * backward_out

        # 应用原始掩码
        output = fused * mask.unsqueeze(-1)

        return output


class TransformerModel(nn.Module):
    """
    蛋白质-RNA相互作用的Transformer模型

    特点:
    - 使用线性复杂度的Performer注意力
    - 差分设计计算突变前后的差异
    - 双向处理捕获序列信息
    """

    def __init__(
            self,
            protein_channels,  # 蛋白质输入特征维度
            rna_channels=None,  # RNA输入特征维度（为接口兼容性保留）
            hidden_channels=64,  # 隐藏层维度
            out_channels=1,  # 输出维度
            num_layers=3,  # Transformer层数
            nhead=8,  # 注意力头数
            dropout=0.1,  # Dropout比例
            **kwargs  # 其他参数
    ):
        super().__init__()

        # 保存配置
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        print(f"初始化Performer模型 - 蛋白质特征:{protein_channels}, "
              f"隐藏维度:{hidden_channels}, 层数:{num_layers}, 头数:{nhead}")

        # 蛋白质特征编码器
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )

        # 蛋白质类型嵌入（可学习的参数）
        self.protein_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))

        # 双向Performer
        self.transformer = BidirectionalPerformer(
            d_model=hidden_channels,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_channels * 4,  # 常用比例
            dropout=dropout
        )

        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def encode_protein(self, data):
        """
        编码蛋白质数据

        参数:
            data: 蛋白质图数据

        返回:
            encoded: 编码后的特征
            batch: 批次信息
        """
        # 使用to_dense_batch获取序列表示
        x_dense, mask = to_dense_batch(data.x, data.batch)

        # 编码节点特征
        x_encoded = self.protein_encoder(x_dense)

        # 添加蛋白质类型嵌入
        x_encoded = x_encoded + self.protein_embedding

        # 应用Transformer编码器
        x_transformed = self.transformer(x_encoded, mask)

        return x_transformed, mask, data.batch

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        前向传播函数 - 计算差分预测

        参数:
            wild_data: 野生型蛋白质图数据
            mutant_data: 突变型蛋白质图数据
            rna_data: RNA数据 (可选，本模型不使用)

        返回:
            prediction: 预测的ΔΔG值
        """
        # 编码野生型和突变型蛋白质
        wild_encoded, wild_mask, wild_batch = self.encode_protein(wild_data)
        mutant_encoded, mutant_mask, mutant_batch = self.encode_protein(mutant_data)

        # 提取有效序列特征
        wild_features = wild_encoded[wild_mask]
        mutant_features = mutant_encoded[mutant_mask]

        # 全局池化
        wild_pooled = scatter_mean(wild_features, wild_batch, dim=0)
        mutant_pooled = scatter_mean(mutant_features, mutant_batch, dim=0)

        # 计算差异特征
        diff_features = mutant_pooled - wild_pooled

        # 预测
        prediction = self.predictor(diff_features).squeeze(-1)

        return prediction