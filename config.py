"""Shared configuration for the iSCALE research code."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Configuration used for the manuscript release. Keep this mapping synchronized
# with the released checkpoint and configs/paper_config.yaml.
PAPER_MODEL_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 3,
    "d_state": 32,
    "d_conv": 4,
    "expand": 2,
    "headdim": 16,
    "chunk_size": 32,
    "aux_weight": 0.2,
}


def get_config(node_features: int) -> dict:
    """Return the manuscript model configuration for legacy callers."""
    return {
        "in_channels": node_features,
        "hidden_channels": PAPER_MODEL_CONFIG["hidden_channels"],
        "out_channels": 1,
        "gmb_args": {
            "d_model": PAPER_MODEL_CONFIG["hidden_channels"],
            "d_state": PAPER_MODEL_CONFIG["d_state"],
            "d_conv": PAPER_MODEL_CONFIG["d_conv"],
            "expand": PAPER_MODEL_CONFIG["expand"],
            "use_checkpointing": True,
        },
        "num_layers": PAPER_MODEL_CONFIG["num_layers"],
    }


ENCODING_DIM = 16
BATCH_SIZE = 16
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 300
NUM_LAYERS = PAPER_MODEL_CONFIG["num_layers"]
SEED = 42
SHUFFLE = True
VAL_SPLIT = 0.1
DEFAULT_CONTACT_THRESHOLDS = [8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
DEFAULT_CHUNK_SIZE = PAPER_MODEL_CONFIG["chunk_size"]

USE_SUBGRAPHS = True
NUM_HOPS = 3
NUM_WORKERS = 0
PREFETCH_FACTOR = 1

# Optional legacy feature caches remain configurable for older experiments, but
# they are not part of the manuscript release package.
DEFAULT_ESM2_CACHE_DIR = os.getenv(
    "ISCALE_ESM2_CACHE_DIR", str(PROJECT_ROOT / "data" / "optional_feature_cache")
)
ESM2_FEATURE_DIM = 1280
ESM2_FEATURE_TYPES = [4, 5, 6, 7]

DATA_DIR = Path(os.getenv("ISCALE_DATA_DIR", PROJECT_ROOT / "data" / "processed"))
MODEL_DIR = Path(os.getenv("ISCALE_MODEL_DIR", PROJECT_ROOT / "checkpoints"))
DEFAULT_DATASET_PATH = Path(
    os.getenv("ISCALE_DATA_PATH", DATA_DIR / "protein_rna_dataset.pkl")
)


def import_from_path(file_path: str | os.PathLike, module_name: str | None = None):
    """Import a Python module from an explicit path."""
    import importlib.util

    file_path = os.fspath(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Module file does not exist: {file_path}")
    if module_name is None:
        module_name = Path(file_path).stem

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import specification for {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise ImportError(f"Failed to import {module_name} from {file_path}: {exc}") from exc
