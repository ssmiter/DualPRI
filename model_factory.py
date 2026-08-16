"""Model creation and checkpoint-loading helpers for iSCALE."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import torch

from config import DEFAULT_CHUNK_SIZE, PAPER_MODEL_CONFIG


class ModelFactory:
    """Create supported models through a stable public interface."""

    BUILTIN_MODELS = {
        "iscale": "model.iscale.ISCALE",
        "dualssd": "model.iscale.ISCALE",  # Backward-compatible CLI alias.
    }

    MODEL_PARAMS_MAP = {
        "model.iscale.ISCALE": {
            "protein_channels": "protein_channels",
            "rna_channels": "rna_channels",
            "hidden_channels": "hidden_channels",
            "out_channels": 1,
            "num_layers": PAPER_MODEL_CONFIG["num_layers"],
            "dropout": "dropout",
            "d_state": PAPER_MODEL_CONFIG["d_state"],
            "d_conv": PAPER_MODEL_CONFIG["d_conv"],
            "expand": PAPER_MODEL_CONFIG["expand"],
            "headdim": PAPER_MODEL_CONFIG["headdim"],
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "aux_weight": PAPER_MODEL_CONFIG["aux_weight"],
        }
    }

    @classmethod
    def create_model(cls, model_name: str, **kwargs: Any):
        """Create a model from a built-in name or a fully qualified class path."""
        try:
            if model_name in cls.BUILTIN_MODELS:
                return cls._create_from_path(cls.BUILTIN_MODELS[model_name], **kwargs)
            if "." in model_name:
                return cls._create_from_path(model_name, **kwargs)
            raise ValueError(f"Unknown model: {model_name}")
        except Exception as exc:
            raise ValueError(f"Could not create model '{model_name}': {exc}") from exc

    @classmethod
    def _create_from_path(cls, model_path: str, **kwargs: Any):
        """Instantiate a model from ``package.module.ClassName``."""
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_path), class_name)
            if hasattr(model_class, "create_model"):
                return model_class.create_model(**kwargs)
            return model_class(**cls._map_params(model_path, **kwargs))
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Could not import model '{model_path}': {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Could not instantiate model '{model_path}': {exc}") from exc

    @classmethod
    def _map_params(cls, model_path: str, **kwargs: Any) -> dict[str, Any]:
        """Map public training arguments to a model constructor."""
        param_map = cls.MODEL_PARAMS_MAP.get(model_path)
        if param_map is None:
            return dict(kwargs)

        mapped: dict[str, Any] = {}
        missing: list[str] = []
        for target_name, source in param_map.items():
            if target_name in kwargs:
                mapped[target_name] = kwargs[target_name]
            elif isinstance(source, str):
                if source in kwargs:
                    mapped[target_name] = kwargs[source]
                else:
                    missing.append(source)
            else:
                mapped[target_name] = source

        if missing:
            names = ", ".join(sorted(set(missing)))
            raise ValueError(f"Missing required model arguments: {names}")
        return mapped

    @classmethod
    def get_available_models(cls) -> list[str]:
        """Return the supported public model names."""
        return list(cls.BUILTIN_MODELS)

    @classmethod
    def load_model(cls, model_path: str, model_class=None, **kwargs: Any):
        """Load a model and its state dictionary from a training checkpoint."""
        checkpoint_path = Path(model_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        if model_class is not None:
            model = model_class(**kwargs)
            model.load_state_dict(state_dict)
            return model

        model_info = checkpoint.get("model_info", {})
        qualified_name = cls._qualified_name(model_info)
        if qualified_name is None:
            for filename in ("model_info.json", "model_info.txt"):
                info_path = checkpoint_path.parent / filename
                if info_path.is_file():
                    qualified_name = cls._qualified_name(cls._read_model_info(info_path))
                    if qualified_name:
                        break

        if qualified_name is None:
            raise ValueError(
                "The checkpoint does not identify its model class. Pass model_class "
                "explicitly or provide model_info in the checkpoint directory."
            )

        model = cls._create_from_path(qualified_name, **kwargs)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _qualified_name(model_info: dict[str, Any]) -> str | None:
        name = model_info.get("name")
        module = model_info.get("module")
        return f"{module}.{name}" if name and module else None

    @staticmethod
    def _read_model_info(info_path: Path) -> dict[str, str]:
        if info_path.suffix == ".json":
            with info_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        values: dict[str, str] = {}
        with info_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, separator, value = line.partition(":")
                if separator:
                    values[key.strip()] = value.strip()
        return values


def create_model(model_name: str, **kwargs: Any):
    """Create a supported model."""
    return ModelFactory.create_model(model_name, **kwargs)


def load_model(model_path: str, model_class=None, **kwargs: Any):
    """Load a supported model checkpoint."""
    return ModelFactory.load_model(model_path, model_class, **kwargs)


def get_available_models() -> list[str]:
    """Return the supported public model names."""
    return ModelFactory.get_available_models()
