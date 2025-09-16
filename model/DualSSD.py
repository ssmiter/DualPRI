"""
平衡型共享参数的蛋白质-RNA相互作用模型
- 严格按照CoM_SSD_RNA_Ablation实现差分计算
- 平衡辅助任务与主任务，避免喧宾夺主
- RNA使用独立参数处理，尊重数据特性
- 保持与训练器的完美兼容性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add, scatter_mean

from config import DEFAULT_CONTACT_THRESHOLDS
# 从mamba导入SSD函数
from mamba.mamba_ssm.ops.triton.ssd_combined_with_state import mamba_chunk_scan_combined


class BlockContactPredictor(nn.Module):
    """块级接触预测模块：预测接触分布和强度"""

    def __init__(self, d_inner, num_classes=7, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        # 接触分布预测器
        self.contact_distribution = nn.Sequential(
            nn.Linear(d_inner, d_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, num_classes)
        )

        # 接触强度预测器
        self.contact_intensity = nn.Sequential(
            nn.Linear(d_inner, d_inner // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, block_states):
        """预测接触分布和强度"""
        contact_logits = self.contact_distribution(block_states)
        intensity = self.contact_intensity(block_states)
        return contact_logits, intensity


class StructureAwareSSDBlock(nn.Module):
    """结构感知SSD块，能够返回块级状态和块级接触预测"""

    def __init__(
            self,
            d_model,  # 模型维度
            d_state=16,  # 状态维度
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展比例
            headdim=16,  # 头维度
            ngroups=1,  # 组数量
            chunk_size=32,  # 块大小
            num_contact_classes=7,  # 接触分布类别数
            dropout=0.1,  # Dropout比例
            use_contact_pred=True  # 是否使用接触预测
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups
        self.use_contact_pred = use_contact_pred

        # 验证维度兼容性
        assert self.d_inner % self.headdim == 0, "d_inner必须能被headdim整除"
        assert self.nheads % ngroups == 0, "nheads必须能被ngroups整除"

        # 1. 统一投影 - [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj)

        # 2. 卷积层 - 用于局部上下文建模
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,  # 深度可分离卷积
            padding=d_conv - 1,
        )

        # 3. SSM核心参数
        A_log = torch.randn(self.nheads) - 1.0
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.dt_bias._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        # 4. 输出层
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # 5. 块级接触预测模块 (如果启用)
        if use_contact_pred:
            self.contact_predictor = BlockContactPredictor(
                d_inner=self.d_inner,
                num_classes=num_contact_classes,
                dropout=dropout
            )

    def forward(self, x, mask=None):
        """
        前向传播函数
        x: [batch, seq_len, d_model] - 输入序列
        mask: [batch, seq_len] - 序列掩码

        返回:
        output: [batch, seq_len, d_model] - 输出序列
        contact_logits: 接触分布预测或None
        contact_intensities: 接触强度预测或None
        states: [batch, chunks, heads, headdim, dstate] - 块级状态
        """
        batch, seq_len, _ = x.shape

        # 应用掩码
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # 1. 输入投影
        zxbcdt = self.in_proj(x)

        # 2. 分离组件
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # 3. 卷积处理
        xBC = xBC.transpose(1, 2)
        xBC = self.conv1d(xBC)[:, :, :seq_len]
        xBC = xBC.transpose(1, 2)
        xBC = F.silu(xBC)

        # 4. 分离 x, B, C
        x_ssm, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1
        )

        # 5. 重组维度
        x_ssm = rearrange(x_ssm, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)

        # 6. 调用优化的 SSD 计算库
        A = -torch.exp(self.A_log)  # 转换为实际的A参数

        # 获取块级状态信息
        out, final_states, states = mamba_chunk_scan_combined(
            x_ssm, dt, A, B, C,
            chunk_size=self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            return_final_states=True,
            return_states=True  # 启用states返回
        )

        # 7. 转换维度并应用门控
        out = rearrange(out, "b l h p -> b l (h p)")
        out = out * F.silu(z)  # 应用门控

        # 8. 输出处理
        out = self.norm(out)
        out = self.out_proj(out)

        # 9. 块级接触预测（如果启用）
        contact_logits = None
        contact_intensities = None

        if self.use_contact_pred:
            # 提取块级状态信息用于接触预测
            chunk_states = states.mean(dim=3).view(batch, -1, self.d_inner)
            flat_states = chunk_states.reshape(-1, self.d_inner)
            contact_logits, contact_intensities = self.contact_predictor(flat_states)

        return out, contact_logits, contact_intensities, states


class BidirectionalSSDLayer(nn.Module):
    """
    双向结构感知SSD层，结合正向和反向信息
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=16,
                 chunk_size=32, num_contact_classes=7, dropout=0.1, use_contact_pred=True):
        super().__init__()
        assert (d_model * expand) % headdim == 0, "d_model * expand 必须能被 headdim 整除"

        # 前向SSD
        self.forward_ssd = StructureAwareSSDBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            dropout=dropout,
            use_contact_pred=use_contact_pred
        )

        # 后向SSD
        self.backward_ssd = StructureAwareSSDBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            dropout=dropout,
            use_contact_pred=use_contact_pred
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        """
        前向传播
        x: 输入特征 [batch_size, seq_len, hidden_dim]
        mask: 序列掩码 [batch_size, seq_len]

        返回:
        out: 输出特征 [batch_size, seq_len, hidden_dim]
        contacts: 接触分布预测列表
        intensities: 接触强度预测列表
        """
        # 前向SSD处理
        forward_out, fwd_contacts, fwd_intensities, fwd_states = self.forward_ssd(x, mask)

        # 初始化输出列表
        contacts = []
        intensities = []

        # 收集前向预测
        if fwd_contacts is not None:
            contacts.append(fwd_contacts)
        if fwd_intensities is not None:
            intensities.append(fwd_intensities)

        # 后向SSD处理
        x_reversed = torch.flip(x, dims=[1])
        if mask is not None:
            mask_reversed = torch.flip(mask, dims=[1])
        else:
            mask_reversed = None

        backward_out, bwd_contacts, bwd_intensities, bwd_states = self.backward_ssd(x_reversed, mask_reversed)
        backward_out = torch.flip(backward_out, dims=[1])

        # 收集后向预测
        if bwd_contacts is not None:
            contacts.append(bwd_contacts)
        if bwd_intensities is not None:
            intensities.append(bwd_intensities)

        # 门控融合
        combined = torch.cat([forward_out, backward_out], dim=-1)
        gate = self.gate(combined)
        output = gate * forward_out + (1 - gate) * backward_out

        return output, contacts, intensities


class SSD_RNA_Interaction(nn.Module):
    """
    平衡型蛋白质-RNA相互作用预测模型
    兼顾辅助任务与差分DDG计算，RNA独立编码
    """

    def __init__(
            self,
            protein_channels,  # 蛋白质特征维度
            rna_channels,      # RNA特征维度
            hidden_channels,   # 隐藏层维度
            out_channels=1,    # 输出维度
            num_layers=3,      # SSD层数
            d_state=16,        # 状态维度
            d_conv=4,          # 卷积核大小
            expand=2,          # 扩展比例
            headdim=16,        # 头维度
            chunk_size=32,     # 块大小
            num_contact_classes=7, # 接触分布类别数
            contact_thresholds=None, # 接触阈值
            dropout=0.1,       # Dropout比例
            aux_weight=0.1,    # 辅助任务权重
            **kwargs          # 其他参数
    ):
        super().__init__()

        # 保存设置
        # if contact_thresholds is None:
        #     contact_thresholds = [8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

        # 保存设置
        if contact_thresholds is None:
            contact_thresholds = DEFAULT_CONTACT_THRESHOLDS
        # 动态计算类别数 = 阈值数 + 1
        num_contact_classes = len(contact_thresholds) + 1

        self.hidden_dim = hidden_channels
        self.chunk_size = chunk_size
        self.num_contact_classes = num_contact_classes
        self.contact_thresholds = contact_thresholds
        self.aux_weight = aux_weight
        self.gmb_args = {
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand,
            'headdim': headdim,
            'chunk_size': chunk_size
        }

        # 类型编码嵌入
        self.protein_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.rna_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_channels))

        # === 蛋白质处理部分 ===
        # 蛋白质特征编码
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )

        # === RNA处理部分（独立参数）===
        # RNA特征编码
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU()
        )

        # RNA专用SSD层（不共享参数）
        self.rna_ssd_layers = nn.ModuleList([
            BidirectionalSSDLayer(
                d_model=hidden_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                num_contact_classes=num_contact_classes,
                dropout=dropout,
                use_contact_pred=False  # RNA不需要接触预测
            ) for _ in range(num_layers)
        ])

        # === 辅助任务部分 ===
        # 用于辅助任务的蛋白质SSD层（共享参数）
        self.protein_ssd_layers = nn.ModuleList([
            BidirectionalSSDLayer(
                d_model=hidden_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                num_contact_classes=num_contact_classes,
                dropout=dropout,
                use_contact_pred=True  # 启用接触预测
            ) for _ in range(num_layers)
        ])

        # === 交互与预测部分 ===
        # 交互层
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )

        # DDG预测层
        self.ddg_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        前向传播函数，兼容有无RNA的情况

        参数:
            wild_data: 野生型蛋白质数据
            mutant_data: 突变型蛋白质数据
            rna_data: RNA数据（可选）

        返回:
            输出字典，包含预测和中间结果
        """
        # 统一处理输出字典
        outputs = {}

        # 预处理蛋白质数据
        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)

        # 获取序列长度
        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape

        # === 蛋白质特征编码 ===
        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding

        # 应用掩码
        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)

        # === 差分DDG计算路径（类似CoM_SSD_RNA_Ablation）===
        # 差分处理：连接蛋白质序列
        forward_combined = torch.cat([wild_features, mutant_features], dim=1)
        forward_combined_mask = torch.cat([wild_mask, mutant_mask], dim=1)

        backward_combined = torch.cat([mutant_features, wild_features], dim=1)
        backward_combined_mask = torch.cat([mutant_mask, wild_mask], dim=1)

        # 应用SSD层
        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []

        for layer in self.protein_ssd_layers:
            # 正向处理
            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)

            # 反向处理
            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)

        # 提取最终特征
        mutant_forward_features = forward_features[:, wild_len:, :]
        mutant_forward_mask = forward_combined_mask[:, wild_len:]

        wild_backward_features = backward_features[:, mutant_len:, :]
        wild_backward_mask = backward_combined_mask[:, mutant_len:]

        # 安全的全局池化实现 - 避免 batch.repeat 带来的问题
        # 1. 使用标准的池化方法，基于掩码
        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )

        # 计算差异并预测差分DDG
        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)

        # 保存差分DDG和接触预测
        outputs['ddg'] = ddg_diff
        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities

        # === 如果有RNA数据，处理辅助任务和RNA特征 ===
        if rna_data is not None:
            # 预处理RNA数据
            rna_x, rna_mask = to_dense_batch(rna_data.x, rna_data.batch)
            rna_features = self.rna_encoder(rna_x) + self.rna_type_embedding
            rna_features = rna_features * rna_mask.unsqueeze(-1)

            # 分别处理蛋白质辅助任务
            wild_encoded = wild_features
            all_wild_contacts = []
            all_wild_intensities = []

            for layer in self.protein_ssd_layers:
                wild_encoded, contacts, intensities = layer(wild_encoded, wild_mask)
                all_wild_contacts.extend(contacts if contacts else [])
                all_wild_intensities.extend(intensities if intensities else [])

            # 独立处理RNA序列
            rna_encoded = rna_features
            for layer in self.rna_ssd_layers:
                rna_encoded, _, _ = layer(rna_encoded, rna_mask)

            # 保存辅助任务结果
            outputs['wild_contacts'] = all_wild_contacts
            outputs['wild_intensities'] = all_wild_intensities

        return outputs

    def _safe_global_pooling(self, features, mask, batch_indices):
        """
        安全的全局池化实现方法，避免维度不匹配问题

        参数:
            features: [batch, seq_len, hidden_dim] 特征张量
            mask: [batch, seq_len] 掩码张量
            batch_indices: [batch] 批次索引

        返回:
            pooled: [batch, hidden_dim] 池化后的特征
        """
        # 方法1: 使用平均池化
        sum_features = torch.sum(features * mask.unsqueeze(-1), dim=1)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = sum_features / mask_sum

        # 确保输出与批次索引匹配
        batch_size = batch_indices.max().item() + 1
        if pooled.size(0) != batch_size:
            # 如果大小不匹配，使用更安全的方法重新创建结果
            result = torch.zeros(batch_size, self.hidden_dim, device=features.device)
            for b in range(pooled.size(0)):
                if b < batch_size:
                    result[b] = pooled[b]
            pooled = result

        return pooled

    def compute_loss(self, wild_data, mutant_data, rna_data, ddg_target):
        """
        计算损失函数，包括DDG损失和辅助接触预测损失

        参数:
            wild_data: 野生型蛋白质数据
            mutant_data: 突变型蛋白质数据
            rna_data: RNA数据
            ddg_target: DDG目标值

        返回:
            total_loss: 总损失
            loss_dict: 损失组件字典
        """
        # 1. 获取模型预测
        outputs = self.forward(wild_data, mutant_data, rna_data)

        # 2. DDG预测损失
        ddg_loss = F.mse_loss(outputs['ddg'], ddg_target)
        loss_dict = {'ddg_loss': ddg_loss}

        # 3. 辅助任务：接触预测损失
        aux_loss = 0

        if hasattr(wild_data, 'block_contact_dist') and hasattr(wild_data, 'block_contact_int'):
            try:
                # 将目标移到相同设备
                gt_dist = wild_data.block_contact_dist.to(ddg_loss.device)
                gt_int = wild_data.block_contact_int.to(ddg_loss.device)

                # 初始化损失
                contact_loss = 0
                intensity_loss = 0
                valid_predictions = 0

                # 如果有专门的野生型接触预测
                contact_source = (outputs.get('wild_contacts') or
                                  outputs.get('forward_contacts') or [])
                intensity_source = (outputs.get('wild_intensities') or
                                   outputs.get('forward_intensities') or [])

                # 计算每个接触预测的损失
                for contacts, intensities in zip(contact_source, intensity_source):
                    # 跳过不匹配的预测
                    if contacts.size(-1) != gt_dist.size(1):
                        continue

                    # 截取共同的块数
                    min_blocks = min(contacts.size(0), gt_dist.size(0))

                    # 计算KL散度
                    dist_loss = F.kl_div(
                        F.log_softmax(contacts[:min_blocks], dim=-1),
                        gt_dist[:min_blocks],
                        reduction='batchmean'
                    )

                    # 计算接触强度损失
                    int_loss = F.binary_cross_entropy(
                        intensities[:min_blocks],
                        gt_int[:min_blocks],
                        reduction='mean'
                    )

                    # 累加损失
                    contact_loss += dist_loss
                    intensity_loss += int_loss
                    valid_predictions += 1

                # 计算平均损失
                if valid_predictions > 0:
                    contact_loss /= valid_predictions
                    intensity_loss /= valid_predictions

                    # 总辅助损失
                    aux_loss = contact_loss + intensity_loss

                    # 记录损失组件
                    loss_dict['contact_loss'] = contact_loss
                    loss_dict['intensity_loss'] = intensity_loss

            except Exception as e:
                print(f"计算辅助损失时出错: {str(e)}")

        # 4. 计算总损失
        total_loss = ddg_loss + self.aux_weight * aux_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['aux_loss'] = aux_loss

        return total_loss, loss_dict


class SSD_RNA_Ablation(SSD_RNA_Interaction):
    """
    继承自平衡型SSD交互模型的RNA消融版本
    - 接受三元组输入但实际上忽略RNA数据
    - 仅使用差分计算进行DDG预测
    - 保持参数共享模式以支持辅助任务
    """

    def __init__(
            self,
            protein_channels,  # 蛋白质特征维度
            rna_channels,  # RNA特征维度（为了接口兼容）
            hidden_channels,  # 隐藏层维度
            out_channels=1,  # 输出维度
            num_layers=3,  # SSD层数
            d_state=16,  # 状态维度
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展比例
            headdim=16,  # 头维度
            chunk_size=32,  # 块大小
            num_contact_classes=7,  # 接触分布类别数
            contact_thresholds=None,  # 接触阈值
            dropout=0.1,  # Dropout比例
            aux_weight=0.1,  # 辅助任务权重
            **kwargs  # 其他参数
    ):
        # 调用父类构造函数
        super().__init__(
            protein_channels=protein_channels,
            rna_channels=rna_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            contact_thresholds=contact_thresholds,
            dropout=dropout,
            aux_weight=aux_weight,
            **kwargs
        )
        print("初始化平衡型RNA消融模型 - 此模型将接收但忽略RNA数据")

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        重写前向传播函数，忽略RNA数据
        实际上只使用差分计算路径，但保持接口兼容性

        参数:
            wild_data: 野生型蛋白质数据
            mutant_data: 突变型蛋白质数据
            rna_data: RNA数据（将被忽略）

        返回:
            outputs: 输出字典，包含DDG预测和接触信息
        """
        # 统一处理输出字典
        outputs = {}

        # 预处理蛋白质数据
        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)

        # 获取序列长度
        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape

        # === 蛋白质特征编码 ===
        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding

        # 应用掩码
        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)

        # === 仅进行差分DDG计算（类似CoM_SSD_RNA_Ablation）===
        # 连接蛋白质序列
        forward_combined = torch.cat([wild_features, mutant_features], dim=1)
        forward_combined_mask = torch.cat([wild_mask, mutant_mask], dim=1)

        backward_combined = torch.cat([mutant_features, wild_features], dim=1)
        backward_combined_mask = torch.cat([mutant_mask, wild_mask], dim=1)

        # 应用SSD层
        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []

        for layer in self.protein_ssd_layers:
            # 正向处理
            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)

            # 反向处理
            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)

        # 提取最终特征
        mutant_forward_features = forward_features[:, wild_len:, :]
        mutant_forward_mask = forward_combined_mask[:, wild_len:]

        wild_backward_features = backward_features[:, mutant_len:, :]
        wild_backward_mask = backward_combined_mask[:, mutant_len:]

        # 安全的全局池化
        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )

        # 计算差异并预测差分DDG
        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)

        # 保存输出 - 注意我们仍然将forward_contacts作为wild_contacts以支持辅助任务
        outputs['ddg'] = ddg_diff
        outputs['wild_contacts'] = all_forward_contacts  # 用于辅助任务兼容
        outputs['wild_intensities'] = all_forward_intensities  # 用于辅助任务兼容
        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities

        return outputs


class DualSSD(SSD_RNA_Interaction):
    """
    继承自平衡型SSD交互模型的RNA消融版本
    - 接受三元组输入但实际上忽略RNA数据
    - 仅使用差分计算进行DDG预测
    - 保持参数共享模式以支持辅助任务
    - 修复: 解决块对齐问题，确保块不跨越序列边界
    """

    def __init__(
            self,
            protein_channels,  # 蛋白质特征维度
            rna_channels,  # RNA特征维度（为了接口兼容）
            hidden_channels,  # 隐藏层维度
            out_channels=1,  # 输出维度
            num_layers=3,  # SSD层数
            d_state=16,  # 状态维度
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展比例
            headdim=16,  # 头维度
            chunk_size=32,  # 块大小
            num_contact_classes=7,  # 接触分布类别数
            contact_thresholds=None,  # 接触阈值
            dropout=0.1,  # Dropout比例
            aux_weight=0.5,  # 辅助任务权重
            **kwargs  # 其他参数
    ):
        # 调用父类构造函数
        super().__init__(
            protein_channels=protein_channels,
            rna_channels=rna_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            num_contact_classes=num_contact_classes,
            contact_thresholds=contact_thresholds,
            dropout=dropout,
            aux_weight=aux_weight,
            **kwargs
        )
        print("初始化平衡型RNA消融模型 - 此模型将接收但忽略RNA数据")
        self.chunk_size = chunk_size  # 保存块大小用于对齐计算

    def forward(self, wild_data, mutant_data, rna_data=None):
        """
        重写前向传播函数，忽略RNA数据
        实际上只使用差分计算路径，但保持接口兼容性
        修复: 解决块对齐问题，确保块不跨越序列边界

        参数:
            wild_data: 野生型蛋白质数据
            mutant_data: 突变型蛋白质数据
            rna_data: RNA数据（将被忽略）

        返回:
            outputs: 输出字典，包含DDG预测和接触信息
        """
        # 统一处理输出字典
        outputs = {}

        # 预处理蛋白质数据
        wild_x, wild_mask = to_dense_batch(wild_data.x, wild_data.batch)
        mutant_x, mutant_mask = to_dense_batch(mutant_data.x, mutant_data.batch)

        # 获取序列长度
        batch_size, wild_len, _ = wild_x.shape
        _, mutant_len, _ = mutant_x.shape

        # === 蛋白质特征编码 ===
        wild_features = self.protein_encoder(wild_x) + self.protein_type_embedding
        mutant_features = self.protein_encoder(mutant_x) + self.protein_type_embedding

        # 应用掩码
        wild_features = wild_features * wild_mask.unsqueeze(-1)
        mutant_features = mutant_features * mutant_mask.unsqueeze(-1)

        # === 块对齐处理 ===
        # 计算填充所需的长度，使序列长度是chunk_size的倍数
        wild_padded_len = ((wild_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        wild_padding = wild_padded_len - wild_len

        if wild_padding > 0:
            padding_features = torch.zeros(batch_size, wild_padding, self.hidden_dim,
                                           device=wild_features.device)
            padding_mask = torch.zeros(batch_size, wild_padding,
                                       device=wild_mask.device)

            wild_features_padded = torch.cat([wild_features, padding_features], dim=1)
            wild_mask_padded = torch.cat([wild_mask, padding_mask], dim=1)
        else:
            wild_features_padded = wild_features
            wild_mask_padded = wild_mask

        # 同样处理突变型序列
        mutant_padded_len = ((mutant_len + self.chunk_size - 1) // self.chunk_size) * self.chunk_size
        mutant_padding = mutant_padded_len - mutant_len

        if mutant_padding > 0:
            padding_features = torch.zeros(batch_size, mutant_padding, self.hidden_dim,
                                           device=mutant_features.device)
            padding_mask = torch.zeros(batch_size, mutant_padding,
                                       device=mutant_mask.device)

            mutant_features_padded = torch.cat([mutant_features, padding_features], dim=1)
            mutant_mask_padded = torch.cat([mutant_mask, padding_mask], dim=1)
        else:
            mutant_features_padded = mutant_features
            mutant_mask_padded = mutant_mask

        # === 连接填充后的序列 ===
        forward_combined = torch.cat([wild_features_padded, mutant_features_padded], dim=1)
        forward_combined_mask = torch.cat([wild_mask_padded, mutant_mask_padded], dim=1)

        backward_combined = torch.cat([mutant_features_padded, wild_features_padded], dim=1)
        backward_combined_mask = torch.cat([mutant_mask_padded, wild_mask_padded], dim=1)

        # 保存原始长度和填充后的长度，用于后续操作
        wild_padded_len = wild_features_padded.size(1)
        mutant_padded_len = mutant_features_padded.size(1)

        # 应用SSD层
        forward_features = forward_combined
        backward_features = backward_combined

        all_forward_contacts = []
        all_forward_intensities = []
        all_backward_contacts = []
        all_backward_intensities = []

        # 直接处理野生型单独的序列，用于生成干净的辅助任务预测
        # 这样就避免了从混合序列中提取接触预测的复杂度
        wild_only_features = wild_features_padded
        wild_only_mask = wild_mask_padded
        wild_only_contacts = []
        wild_only_intensities = []

        # 主模型SSD层处理
        for layer in self.protein_ssd_layers:
            # 差分路径的正向处理
            forward_features, contacts_f, intensities_f = layer(forward_features, forward_combined_mask)
            forward_features = forward_features * forward_combined_mask.unsqueeze(-1)

            if contacts_f:
                all_forward_contacts.extend(contacts_f)
            if intensities_f:
                all_forward_intensities.extend(intensities_f)

            # 差分路径的反向处理
            backward_features, contacts_b, intensities_b = layer(backward_features, backward_combined_mask)
            backward_features = backward_features * backward_combined_mask.unsqueeze(-1)

            if contacts_b:
                all_backward_contacts.extend(contacts_b)
            if intensities_b:
                all_backward_intensities.extend(intensities_b)

            # 野生型单独处理 - 用于干净的接触预测
            wild_only_features, contacts_w, intensities_w = layer(wild_only_features, wild_only_mask)
            wild_only_features = wild_only_features * wild_only_mask.unsqueeze(-1)

            if contacts_w:
                wild_only_contacts.extend(contacts_w)
            if intensities_w:
                wild_only_intensities.extend(intensities_w)

        # 提取最终特征（使用原始长度进行截取）
        mutant_forward_features = forward_features[:, wild_padded_len:wild_padded_len + mutant_len, :]
        mutant_forward_mask = forward_combined_mask[:, wild_padded_len:wild_padded_len + mutant_len]

        wild_backward_features = backward_features[:, mutant_padded_len:mutant_padded_len + wild_len, :]
        wild_backward_mask = backward_combined_mask[:, mutant_padded_len:mutant_padded_len + wild_len]

        # 安全的全局池化
        mutant_forward_pooled = self._safe_global_pooling(
            mutant_forward_features,
            mutant_forward_mask,
            mutant_data.batch
        )

        wild_backward_pooled = self._safe_global_pooling(
            wild_backward_features,
            wild_backward_mask,
            wild_data.batch
        )

        # 计算差异并预测差分DDG
        diff_features = mutant_forward_pooled - wild_backward_pooled
        ddg_diff = self.ddg_predictor(diff_features).squeeze(-1)

        # 保存输出结果
        outputs['ddg'] = ddg_diff
        # 使用野生型单独处理的接触预测进行辅助任务
        outputs['wild_contacts'] = wild_only_contacts
        outputs['wild_intensities'] = wild_only_intensities
        # 保存差分路径的接触预测（用于调试和分析）
        outputs['forward_contacts'] = all_forward_contacts
        outputs['forward_intensities'] = all_forward_intensities
        outputs['backward_contacts'] = all_backward_contacts
        outputs['backward_intensities'] = all_backward_intensities

        return outputs

    def compute_loss(self, wild_data, mutant_data, rna_data, ddg_target):
        """
        计算损失函数，简化版只保留基本的辅助损失计算

        参数:
            wild_data: 野生型蛋白质数据
            mutant_data: 突变型蛋白质数据
            rna_data: RNA数据（将被忽略）
            ddg_target: DDG目标值

        返回:
            total_loss: 总损失
            loss_dict: 损失组件字典
        """
        # 获取模型预测
        outputs = self.forward(wild_data, mutant_data, rna_data)

        # DDG预测损失
        ddg_loss = F.mse_loss(outputs['ddg'], ddg_target)
        loss_dict = {'ddg_loss': ddg_loss}

        # 辅助任务：接触预测损失
        aux_loss = 0

        if hasattr(wild_data, 'block_contact_dist') and hasattr(wild_data, 'block_contact_int'):
            try:
                # 获取真实标签
                gt_dist = wild_data.block_contact_dist.to(ddg_loss.device)
                gt_int = wild_data.block_contact_int.to(ddg_loss.device)

                # 初始化损失
                contact_loss = 0
                intensity_loss = 0
                valid_predictions = 0

                # 使用专用于野生型的接触预测
                contact_source = outputs.get('wild_contacts', [])
                intensity_source = outputs.get('wild_intensities', [])

                # 针对每个预测计算损失
                for contacts, intensities in zip(contact_source, intensity_source):
                    # 检查尺寸兼容性
                    if contacts.size(-1) != gt_dist.size(1):
                        print(f"Contact prediction size mismatch: {contacts.size()} vs {gt_dist.size()}")
                        continue

                    # 截取共同的块数
                    min_blocks = min(contacts.size(0), gt_dist.size(0))

                    # 基本KL散度损失
                    dist_loss = F.kl_div(
                        F.log_softmax(contacts[:min_blocks], dim=-1),
                        gt_dist[:min_blocks],
                        reduction='batchmean'
                    )

                    # 基本二元交叉熵损失
                    int_loss = F.binary_cross_entropy(
                        intensities[:min_blocks],
                        gt_int[:min_blocks],
                        reduction='mean'
                    )

                    # 累加损失
                    contact_loss += dist_loss
                    intensity_loss += int_loss
                    valid_predictions += 1

                # 计算平均损失
                if valid_predictions > 0:
                    contact_loss /= valid_predictions
                    intensity_loss /= valid_predictions

                    # 总辅助损失
                    aux_loss = contact_loss + intensity_loss

                    # 记录损失组件
                    loss_dict['contact_loss'] = contact_loss
                    loss_dict['intensity_loss'] = intensity_loss

            except Exception as e:
                print(f"计算辅助损失时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 计算总损失
        total_loss = ddg_loss + self.aux_weight * aux_loss
        loss_dict['total_loss'] = total_loss
        loss_dict['aux_loss'] = aux_loss

        return total_loss, loss_dict
