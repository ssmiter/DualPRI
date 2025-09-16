#!/usr/bin/env python3
"""
蛋白质-RNA相互作用预测模型训练和评估主脚本
"""
import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime

from config import SHUFFLE, DEFAULT_CHUNK_SIZE
from utils import calculate_feature_dimensions, get_feature_type_name
from model.utils.loader.enhanced_contact_data_loader import load_protein_rna_data, DEFAULT_CONTACT_THRESHOLDS
from model_factory import create_model, load_model
from trainer import train_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='蛋白质-RNA相互作用预测模型训练和评估')

    # 数据相关参数
    parser.add_argument('--data_path', type=str, default="./dataset_process/dataset/protein_rna_dataset.pkl",
                        help='数据集路径，例如 ./dataset_process/dataset/enhanced_protein_rna_dataset.pkl')
    parser.add_argument('--feature_type', type=int, default=0, choices=[0, 1, 2, 3],
                        help='多尺度特征类型: 0=无特征, 1=仅分布特征, 2=仅强度特征, 3=完整特征')
    parser.add_argument('--force_recompute', action='store_true', default=False,
                        help='强制重新计算接触特征而非使用缓存')
    parser.add_argument('--cache_dir', type=str, default="./contact_cache",
                        help='接触特征缓存目录路径')
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help='SSD块大小，也用于接触特征计算')
    parser.add_argument('--val_ratio', type=float, default=0.1,  # 0.15
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,  # 0.3
                        help='测试集比例')
    parser.add_argument('--batch_size', type=int, default=16,  # for ssd_rna:16
                        help='批次大小')
    parser.add_argument('--no_reverse', default=False,
                        action='store_true', help='不添加反向突变样本')
    parser.add_argument('--train_ratio', type=float, default=None,
                        help='Training set ratio (used with train_ratio split strategy)')
    parser.add_argument('--split_strategy', type=str, default='random',
                        choices=['random', 'pdb_limited', 'train_ratio'], help='pdb_limited: 2 samples per PDB')
    # best for ssd_rna batch=16, lr=0.0008
    # GNN特定参数
    parser.add_argument('--heads', type=int, default=4,
                        help='GAT模型的注意力头数')

    # 模型相关参数
    parser.add_argument('--model', type=str, default='dualssd',
                        choices=['dualssd', 'gcn', 'gat', 'sage',
                                 'gin', 'edge', 'transformer', 'graph_transformer'],
                        help='模型名称，包括SSD变体和对比模型：基础GNN模型以及Transformer类模型')

    parser.add_argument('--protein_channels', type=int, default=41,  # 41
                        help='蛋白质特征通道数')
    parser.add_argument('--rna_channels', type=int, default=5,  # 5
                        help='RNA特征通道数')
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='隐藏层通道数')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout比例')


    # 训练相关参数
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='指定使用的GPU ID')
    parser.add_argument('--learning_rate', type=float, default=0.0008,  # for ssd_rna:0.0008
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
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                        help='检查点保存间隔')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='从检查点恢复训练')

    # 输出相关参数
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='实验名称')

    # 测试相关参数
    parser.add_argument('--test_only', action='store_true',
                        help='仅进行测试，不训练模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='测试的模型路径')

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
        # device = torch.device('cuda')
        device = torch.device(f'cuda:{args.gpu_id}')  # 使用指定的GPU ID

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 在特征类型和时间戳之间添加描述性标签
    feature_type_labels = {0: "base", 1: "dist", 2: "int", 3: "full"}
    feature_label = feature_type_labels[args.feature_type]

    if args.experiment_name:
        output_dir = os.path.join(args.output_dir, f"{args.experiment_name}_{feature_label}_{timestamp}")
    else:
        output_dir = os.path.join(args.output_dir, f"{args.model}_{feature_label}_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    return device, output_dir


def save_config(args, output_dir):
    """保存实验配置"""
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def main():
    args = parse_args()

    # 设置环境
    device, output_dir = setup_environment(args)

    # 保存配置
    save_config(args, output_dir)

    print(f"使用设备: {device}")
    print(f"输出目录: {output_dir}")

    # 使用工具函数计算特征维度
    base_protein_channels = 41  # 原始蛋白质特征维度
    base_rna_channels = 5  # 原始RNA特征维度

    # 计算正确的特征维度
    protein_channels, rna_channels = calculate_feature_dimensions(
        base_protein_channels=base_protein_channels,
        base_rna_channels=base_rna_channels,
        feature_type=args.feature_type,
        contact_thresholds=DEFAULT_CONTACT_THRESHOLDS
    )

    # 打印特征类型和维度信息
    print(f"特征类型: {get_feature_type_name(args.feature_type)}")

    if args.feature_type > 0:
        print(f"使用合并特征 - 蛋白质通道数: {protein_channels}, RNA通道数: {rna_channels}")
    else:
        print(f"不使用合并特征 - 蛋白质通道数: {protein_channels}, RNA通道数: {rna_channels}")

    # 覆盖命令行参数
    args.protein_channels = protein_channels
    args.rna_channels = rna_channels

    # 加载数据
    print("加载数据集...")
    print("shuffle:", SHUFFLE)
    train_loader, val_loader, test_loader = load_protein_rna_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        add_reverse=not args.no_reverse,
        seed=args.seed,
        split_strategy=args.split_strategy,
        feature_type=args.feature_type,  # 传递特征类型
        force_recompute=args.force_recompute,
        # cache_dir="./contact_cache",
        cache_dir=args.cache_dir,  # 使用命令行传入的缓存目录
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,  # 添加块大小参数
    )

    # 创建或加载模型
    if args.test_only and args.model_path:
        print(f"加载模型: {args.model_path}")
        model = load_model(args.model_path)
    else:
        print(f"创建模型: {args.model}")
        model = create_model(
            args.model,
            protein_channels=args.protein_channels,
            rna_channels=args.rna_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            # SSD模型特定参数
            chunk_size=args.chunk_size,
        )

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    # 训练或测试
    if args.test_only:
        print("仅进行测试...")
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir
        )
        test_metrics, _, _ = trainer.test()
    else:
        print("开始训练...")
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            max_epochs=args.epochs,
            patience=args.patience,
            checkpoint_interval=args.checkpoint_interval,
            resume_from=args.resume_from
        )

    print(f"所有结果已保存至 {output_dir}")

    # 返回测试结果供批处理脚本使用
    if 'test_metrics' in locals() and test_metrics:
        return test_metrics
    return None


if __name__ == "__main__":
    main()
