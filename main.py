#!/usr/bin/env python3
"""Train or evaluate iSCALE for protein-RNA binding-affinity prediction."""

import argparse
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import DEFAULT_CHUNK_SIZE, DEFAULT_DATASET_PATH, SHUFFLE
from model.utils.loader.enhanced_contact_data_loader import (
    DEFAULT_CONTACT_THRESHOLDS,
    load_protein_rna_data,
)
from model_factory import create_model, load_model
from trainer import train_model
from utils import calculate_feature_dimensions, get_feature_type_name


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate iSCALE for protein-RNA binding-affinity prediction."
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--data_path",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the processed dataset (see docs/DATA.md).",
    )
    data.add_argument(
        "--feature_type",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Feature set: 0=base, 1=distance, 2=intensity, 3=all.",
    )
    data.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute contact features instead of using the cache.",
    )
    data.add_argument(
        "--cache_dir", default="./contact_cache", help="Contact-feature cache directory."
    )
    data.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="SSD chunk size; also used for contact-feature calculation.",
    )
    data.add_argument("--val_ratio", type=float, default=0.1, help="Validation-set ratio.")
    data.add_argument("--test_ratio", type=float, default=0.1, help="Test-set ratio.")
    data.add_argument("--batch_size", type=int, default=16, help="Mini-batch size.")
    data.add_argument(
        "--no_reverse", action="store_true", help="Do not add reverse-mutation samples."
    )
    data.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="Training-set ratio for the train_ratio split strategy.",
    )
    data.add_argument(
        "--split_strategy",
        default="random",
        choices=["random", "pdb_limited", "train_ratio"],
        help="Dataset split strategy; pdb_limited uses two samples per PDB entry.",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        default="iscale",
        choices=["iscale", "dualssd"],
        help='Model name; "dualssd" is retained as a legacy alias.',
    )
    model.add_argument("--protein_channels", type=int, default=41)
    model.add_argument("--rna_channels", type=int, default=5)
    model.add_argument("--hidden_channels", type=int, default=64)
    model.add_argument("--num_layers", type=int, default=3)
    model.add_argument("--dropout", type=float, default=0.1)

    training = parser.add_argument_group("training")
    training.add_argument("--gpu_id", type=int, default=0, help="CUDA device index.")
    training.add_argument("--learning_rate", type=float, default=0.0008)
    training.add_argument("--weight_decay", type=float, default=1e-5)
    training.add_argument("--epochs", type=int, default=300)
    training.add_argument("--patience", type=int, default=40)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--no_cuda", action="store_true", help="Run on the CPU.")
    training.add_argument("--checkpoint_interval", type=int, default=100)
    training.add_argument("--resume_from", default=None, help="Checkpoint used to resume training.")

    output = parser.add_argument_group("output")
    output.add_argument("--output_dir", default="./output")
    output.add_argument("--experiment_name", default="")
    output.add_argument("--test_only", action="store_true", help="Evaluate without training.")
    output.add_argument("--model_path", default=None, help="Checkpoint used for evaluation.")
    return parser.parse_args()


def setup_environment(args):
    """Set random seeds, select the device, and create the output directory."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_label = {0: "base", 1: "dist", 2: "int", 3: "full"}[args.feature_type]
    run_name = args.experiment_name or args.model
    output_dir = os.path.join(args.output_dir, f"{run_name}_{feature_label}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return device, output_dir


def save_config(args, output_dir):
    """Write the resolved run arguments to the output directory."""
    config_path = os.path.join(output_dir, "config.txt")
    with open(config_path, "w", encoding="utf-8") as handle:
        for name, value in vars(args).items():
            handle.write(f"{name}: {value}\n")


def main():
    args = parse_args()

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {data_path}. "
            "Download or build it as described in docs/DATA.md, or pass --data_path."
        )
    if args.test_only and not args.model_path:
        raise ValueError("--test_only requires --model_path.")
    args.data_path = str(data_path)

    device, output_dir = setup_environment(args)
    save_config(args, output_dir)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")

    protein_channels, rna_channels = calculate_feature_dimensions(
        base_protein_channels=41,
        base_rna_channels=5,
        feature_type=args.feature_type,
        contact_thresholds=DEFAULT_CONTACT_THRESHOLDS,
    )
    args.protein_channels = protein_channels
    args.rna_channels = rna_channels
    print(f"Feature set: {get_feature_type_name(args.feature_type)}")
    print(f"Input channels: protein={protein_channels}, RNA={rna_channels}")

    print(f"Loading dataset (shuffle={SHUFFLE})...")
    train_loader, val_loader, test_loader = load_protein_rna_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        add_reverse=not args.no_reverse,
        seed=args.seed,
        split_strategy=args.split_strategy,
        feature_type=args.feature_type,
        force_recompute=args.force_recompute,
        cache_dir=args.cache_dir,
        train_ratio=args.train_ratio,
        chunk_size=args.chunk_size,
    )

    if args.test_only:
        print(f"Loading model: {args.model_path}")
        model = load_model(args.model_path)
    else:
        print(f"Creating model: {args.model}")
        model = create_model(
            args.model,
            protein_channels=args.protein_channels,
            rna_channels=args.rna_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
            chunk_size=args.chunk_size,
        )

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Parameters: total={total_params:,}, trainable={trainable_params:,}")

    if args.test_only:
        trainer = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            output_dir=output_dir,
        )
        test_metrics, _, _ = trainer.test()
    else:
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
            resume_from=args.resume_from,
        )
        test_metrics = None

    print(f"Results saved to {output_dir}")
    return test_metrics


if __name__ == "__main__":
    main()
