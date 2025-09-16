#!/usr/bin/env python3
"""
蛋白质-RNA相互作用预测模型交叉验证脚本
用于执行PDB-based k-fold交叉验证实验
"""
import os
import argparse
import torch
import numpy as np
import random
import json
import pickle
import pandas as pd
from datetime import datetime

from config import SHUFFLE, DEFAULT_CHUNK_SIZE
from utils import (calculate_feature_dimensions, get_feature_type_name,
                   pdb_based_kfold_split, pdb_based_kfold_split_with_randomness, load_fold_splits, create_fold_dataloaders,
                   visualize_cv_results, random_kfold_split, )
from model.utils.loader.enhanced_contact_data_loader import DEFAULT_CONTACT_THRESHOLDS, EnhancedProteinRNADataLoader
from model_factory import create_model
from trainer import train_model, SemiSupervisedProteinRNATrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='蛋白质-RNA相互作用预测模型交叉验证')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default="./dataset_process/dataset/protein_rna_dataset.pkl",
                        help='数据集路径，例如 ./dataset_process/dataset/protein_rna_dataset.pkl')
    parser.add_argument('--feature_type', type=int, default=4, choices=[0, 1, 2, 3],
                        help='多尺度特征类型: 0=无特征, 1=仅分布特征, 2=仅强度特征, 3=全部特征')

    parser.add_argument('--esm2_cache_dir', type=str, default="./dataset_process/esm2_features",
                        help='ESM2特征缓存目录路径')
    parser.add_argument('--check_esm2_features', action='store_true', default=True,
                        help='检查ESM2特征文件的完整性')
    parser.add_argument('--no_check_esm2', action='store_true', default=False,
                        help='跳过ESM2特征完整性检查')
    parser.add_argument('--force_recompute', action='store_true', default=False,
                        help='强制重新计算接触特征而非使用缓存')
    parser.add_argument('--cache_dir', type=str, default="./contact_cache",
                        help='接触特征缓存目录路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--no_reverse', default=False,
                        action='store_true', help='不添加反向突变样本')
    parser.add_argument('--contact_thresholds', type=str, default=None,
                        help='接触距离阈值，逗号分隔，例如: "8.0,10.0,15.0,20.0,30.0,50.0"')

    # 交叉验证相关参数
    parser.add_argument('--k_folds', type=int, default=5,
                        help='交叉验证的折数')
    parser.add_argument('--redivide', action='store_true', default=False,
                        help='强制重新划分数据集，即使已有划分文件')
    parser.add_argument('--split_method', type=str, default='random')
    # 模型相关参数
    parser.add_argument('--model', type=str, default='dualssd',
                        choices=['dualssd', 'gcn', 'gat', 'sage',
                                 'gin', 'edge', 'transformer', 'graph_transformer'],
                        help='模型名称')
    parser.add_argument('--protein_channels', type=int, default=None,  # 41
                        help='蛋白质特征通道数(如果不指定将自动计算)')
    parser.add_argument('--rna_channels', type=int, default=None,  # 5
                        help='RNA特征通道数(如果不指定将自动计算)')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='隐藏层通道数')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比例')

    # 添加SSD模型特定参数
    parser.add_argument('--update_ssd_params', action='store_true',
                        help='更新SSD模型参数映射以支持超参数搜索')
    parser.add_argument('--d_state', type=int, default=32,
                        help='SSD状态维度')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='SSD卷积核大小')
    parser.add_argument('--expand', type=int, default=2,
                        help='SSD扩展比例')
    parser.add_argument('--headdim', type=int, default=16,
                        help='SSD头维度')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='SSD块大小，也用于接触特征计算')
    parser.add_argument('--aux_weight', type=float, default=0.2,
                        help='辅助任务权重（仅用于带辅助任务的模型）')

    # 训练相关参数
    parser.add_argument('--learning_rate', type=float, default=0.0008,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=40,
                        help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_cuda', action='store_true',
                        help='不使用CUDA')
    parser.add_argument('--checkpoint_interval', type=int, default=200,
                        help='检查点保存间隔')

    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='./cv_results',
                        help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='实验名称')

    return parser.parse_args()


def setup_environment(args):
    """设置运行环境"""
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 在特征类型和时间戳之间添加描述性标签
    # 🔥 更新特征类型标签支持0-7
    feature_type_labels = {
        0: "base", 1: "dist", 2: "int", 3: "full",
        4: "esm2", 5: "esm2_dist", 6: "esm2_int", 7: "esm2_full"
    }
    feature_label = feature_type_labels[args.feature_type]

    if args.experiment_name:
        output_dir = os.path.join(args.output_dir,
                                  f"{args.experiment_name}_{feature_label}_seed{args.seed}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.model}_{feature_label}_seed{args.seed}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    return device, output_dir


def save_config(args, output_dir):
    """保存实验配置"""
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def check_esm2_features_availability(esm2_cache_dir, verbose=True):
    """
    检查ESM2特征的可用性

    Args:
        esm2_cache_dir: ESM2特征缓存目录
        verbose: 是否输出详细信息

    Returns:
        bool: 是否可用
    """
    if not os.path.exists(esm2_cache_dir):
        if verbose:
            print(f"❌ ESM2特征目录不存在: {esm2_cache_dir}")
        return False

    # 检查ESM2文件
    esm2_files = [f for f in os.listdir(esm2_cache_dir) if f.endswith('_esm2.pt')]

    if verbose:
        if len(esm2_files) > 0:
            print(f"✅ 发现 {len(esm2_files)} 个ESM2特征文件")
        else:
            print(f"❌ ESM2特征目录中没有特征文件")

    return len(esm2_files) > 0


def is_esm2_feature_type(feature_type):
    """检查是否为ESM2特征类型"""
    return feature_type in [4, 5, 6, 7]
def run_kfold_cross_validation(args, device, output_dir):
    """
    执行k折交叉验证实验

    参数:
        args: 命令行参数
        device: 计算设备
        output_dir: 输出目录

    返回:
        avg_metrics: 平均指标
    """
    # 🔥 添加ESM2特征检查
    if is_esm2_feature_type(args.feature_type):
        print(f"🧬 检测到ESM2特征类型: {get_feature_type_name(args.feature_type)}")

        if not check_esm2_features_availability(args.esm2_cache_dir):
            print(f"❌ 错误: ESM2特征不可用")
            print("请先运行ESM2特征提取脚本:")
            print(f"   python dataset_esm2_simplified.py")
            raise FileNotFoundError(f"ESM2特征不可用: {args.esm2_cache_dir}")

    # 1. 解析接触阈值参数
    contact_thresholds = DEFAULT_CONTACT_THRESHOLDS  # 默认值
    if args.contact_thresholds:
        try:
            contact_thresholds = [float(x.strip()) for x in args.contact_thresholds.split(',')]
            print(f"使用自定义接触阈值: {contact_thresholds}")
        except Exception as e:
            print(f"解析接触阈值参数出错: {str(e)}, 使用默认阈值")
            contact_thresholds = DEFAULT_CONTACT_THRESHOLDS
    else:
        print(f"使用默认接触阈值: {contact_thresholds}")

    # 2. 加载完整数据集...
    print("加载数据集...")

    # 先加载原始数据获取PyG格式数据列表
    with open(args.data_path, 'rb') as f:
        dataset = pickle.load(f)

    # 使用工具函数计算特征维度
    base_protein_channels = 41  # 原始蛋白质特征维度
    base_rna_channels = 5  # 原始RNA特征维度

    # 如果用户没有指定channels，则自动计算
    if args.protein_channels is None or args.rna_channels is None:
        protein_channels, rna_channels = calculate_feature_dimensions(
            base_protein_channels=base_protein_channels,
            base_rna_channels=base_rna_channels,
            feature_type=args.feature_type,
            contact_thresholds=contact_thresholds
        )

        # 更新args
        if args.protein_channels is None:
            args.protein_channels = protein_channels
        if args.rna_channels is None:
            args.rna_channels = rna_channels
    else:
        # 使用用户指定的值
        protein_channels = args.protein_channels
        rna_channels = args.rna_channels

    protein_channels, rna_channels = calculate_feature_dimensions(
        base_protein_channels=base_protein_channels,
        base_rna_channels=base_rna_channels,
        feature_type=args.feature_type,
        contact_thresholds=contact_thresholds  # 使用解析后的阈值
    )

    # 覆盖命令行参数
    args.protein_channels = protein_channels
    args.rna_channels = rna_channels

    # 打印特征类型和维度信息
    print(f"特征类型: {get_feature_type_name(args.feature_type)}")
    print(f"蛋白质通道数: {protein_channels}, RNA通道数: {rna_channels}")
    print(f"接触阈值数量: {len(contact_thresholds)} (类别数: {len(contact_thresholds) + 1})")

    # 加载所有数据，但不拆分
    data_loader = EnhancedProteinRNADataLoader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_ratio=0,  # 不做自动拆分
        test_ratio=0,  # 不做自动拆分
        add_reverse=not args.no_reverse,
        seed=args.seed,
        shuffle=SHUFFLE,
        feature_type=args.feature_type,
        force_recompute=args.force_recompute,
        # cache_dir="./contact_cache",
        cache_dir=args.cache_dir,  # 使用命令行传入的缓存目录
        chunk_size=args.chunk_size,  # 添加块大小参数
        contact_thresholds=contact_thresholds,  # 关键：传入自定义阈值
        esm2_cache_dir=args.esm2_cache_dir,
        check_esm2_features=args.check_esm2_features and not args.no_check_esm2
    )

    # 获取完整的数据列表
    data_list = data_loader.data_list
    print(f"数据集加载完成，共 {len(data_list)} 个样本")

    # 2. 检查是否有保存的划分，如果没有或者要求重新划分，则创建新的划分
    fold_dir = os.path.join(output_dir, 'folds')
    fold_indices_file = os.path.join(fold_dir, 'fold_indices.npz')

    if args.redivide or not os.path.exists(fold_indices_file):
        # 创建k-fold划分，根据参数选择划分方法
        if args.split_method == 'pdb_based':
            # splits = pdb_based_kfold_split(
            splits = pdb_based_kfold_split_with_randomness(
                data_list=data_list,
                k=args.k_folds,
                seed=args.seed,
                output_dir=fold_dir,
                visualize=True
            )
        else:  # random
            splits = random_kfold_split(
                data_list=data_list,
                k=args.k_folds,
                seed=args.seed,
                output_dir=fold_dir,
                visualize=True
            )
    else:
        # 加载现有划分
        splits = load_fold_splits(fold_indices_file)

    # 3. 进行k折交叉验证
    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print(f"\n===== 训练 Fold {fold_idx + 1}/{args.k_folds} =====")

        # 创建当前fold的数据加载器
        train_loader, val_loader = create_fold_dataloaders(
            data_list=data_list,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=args.batch_size
        )

        # 在创建模型前添加（只处理dualssd模型）
        if args.update_ssd_params and args.model == 'dualssd':
            from model_factory import ModelFactory
            model_path = 'model.CoM_SSD_RNA_Interaction_Simplified.CoM_SSD_RNA_Ablation_v2'

            if model_path in ModelFactory.MODEL_PARAMS_MAP:
                param_map = ModelFactory.MODEL_PARAMS_MAP[model_path]
                # 更新参数映射
                param_map['d_state'] = 'd_state'
                param_map['d_conv'] = 'd_conv'
                param_map['expand'] = 'expand'
                param_map['headdim'] = 'headdim'
                param_map['chunk_size'] = 'chunk_size'
                param_map['aux_weight'] = 'aux_weight'
                print(f"已更新{args.model}模型参数映射以支持超参数搜索")


        # 创建模型
        model = create_model(
            args.model,
            protein_channels=args.protein_channels,
            rna_channels=args.rna_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            # SSD模型特定参数
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            headdim=args.headdim,
            chunk_size=args.chunk_size,
            aux_weight=args.aux_weight
        )

        # 打印模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型总参数数量: {total_params:,}")
        print(f"可训练参数数量: {trainable_params:,}")

        # 训练模型
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,  # 在交叉验证中，我们不使用单独的测试集
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=fold_output_dir,
            max_epochs=args.epochs,
            patience=args.patience,
            checkpoint_interval=args.checkpoint_interval
        )

        # 评估该fold上的最佳模型性能
        best_val_metrics = trainer.val_metrics[trainer.best_epoch]

        # 收集预测结果以便绘图
        all_preds = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                wild_data, mutant_data, rna_data, ddg = batch
                wild_data = wild_data.to(device)
                mutant_data = mutant_data.to(device)
                rna_data = rna_data.to(device)

                if isinstance(ddg, (list, tuple)):
                    ddg = ddg[0]

                if isinstance(ddg, torch.Tensor):
                    target = ddg.to(device)
                    if target.dim() > 1:
                        target = target.squeeze()
                else:
                    target = torch.tensor([ddg], dtype=torch.float, device=device)

                output = model(wild_data, mutant_data, rna_data)

                # 检查输出类型
                if isinstance(output, dict):
                    pred = output['ddg']
                elif isinstance(output, torch.Tensor):
                    pred = output
                elif isinstance(output, tuple):
                    pred = output[0]

                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # 保存预测结果以便后续可视化
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        np.save(os.path.join(fold_output_dir, "predictions.npy"),
                {'predictions': all_preds, 'true_values': all_targets})

        # 记录本fold的结果
        fold_result = {
            'fold': fold_idx + 1,
            'best_epoch': trainer.best_epoch + 1,
            'mse': best_val_metrics['mse'],
            'mae': best_val_metrics['mae'],
            'pcc': best_val_metrics['pcc']
        }
        fold_results.append(fold_result)

        print(f"Fold {fold_idx + 1} 最佳结果 (Epoch {fold_result['best_epoch']}):")
        print(f"  MSE: {fold_result['mse']:.4f}")
        print(f"  MAE: {fold_result['mae']:.4f}")
        print(f"  PCC: {fold_result['pcc']:.4f}")

    # 4. 计算并保存平均性能
    avg_metrics = {
        'mse': np.mean([r['mse'] for r in fold_results]),
        'mse_std': np.std([r['mse'] for r in fold_results]),
        'mae': np.mean([r['mae'] for r in fold_results]),
        'mae_std': np.std([r['mae'] for r in fold_results]),
        'pcc': np.mean([r['pcc'] for r in fold_results]),
        'pcc_std': np.std([r['pcc'] for r in fold_results])
    }

    # 5. 保存结果摘要
    with open(os.path.join(output_dir, 'cv_results_summary.txt'), 'w') as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"特征类型: {args.feature_type} ({get_feature_type_name(args.feature_type)})\n")
        f.write(f"K折交叉验证结果 (k={args.k_folds})\n")
        f.write("======================\n\n")

        for fold_result in fold_results:
            f.write(f"Fold {fold_result['fold']}:\n")
            f.write(f"  最佳Epoch: {fold_result['best_epoch']}\n")
            f.write(f"  MSE: {fold_result['mse']:.4f}\n")
            f.write(f"  MAE: {fold_result['mae']:.4f}\n")
            f.write(f"  PCC: {fold_result['pcc']:.4f}\n\n")

        f.write("平均性能:\n")
        f.write(f"  MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}\n")
        f.write(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}\n")
        f.write(f"  PCC: {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}\n")

    # 保存为CSV以便后续分析
    df_results = pd.DataFrame(fold_results)
    df_results.to_csv(os.path.join(output_dir, 'fold_results.csv'), index=False)

    # 6. 可视化结果
    visualize_cv_results(fold_results, avg_metrics, output_dir)

    # 7. 打印最终结果
    print("\n===== 交叉验证最终结果 =====")
    print(f"模型: {args.model}")
    print(f"特征类型: {args.feature_type} ({get_feature_type_name(args.feature_type)})")
    print(f"平均性能:")
    print(f"  MSE: {avg_metrics['mse']:.4f} ± {avg_metrics['mse_std']:.4f}")
    print(f"  MAE: {avg_metrics['mae']:.4f} ± {avg_metrics['mae_std']:.4f}")
    print(f"  PCC: {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}")

    return avg_metrics



def main():
    args = parse_args()

    # 设置运行环境
    device, output_dir = setup_environment(args)

    # 保存更新后的配置（包含ESM2参数）
    config = vars(args).copy()
    config.update({
        'device': str(device),
        'feature_type_name': get_feature_type_name(args.feature_type),
        'is_esm2_feature': is_esm2_feature_type(args.feature_type)
    })

    # 保存配置
    save_config(args, output_dir)

    print(f"使用设备: {device}")
    print(f"输出目录: {output_dir}")

    avg_metrics = run_kfold_cross_validation(args, device, output_dir)

    print(f"交叉验证完成，所有结果已保存至 {output_dir}")
    print(f"最终平均性能: PCC = {avg_metrics['pcc']:.4f} ± {avg_metrics['pcc_std']:.4f}")

    return avg_metrics


if __name__ == "__main__":
    main()
