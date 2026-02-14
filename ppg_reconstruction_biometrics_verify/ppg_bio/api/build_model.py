from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_model_from_config(cfg: Dict[str, Any]):
    """Build model object from config.

    Note: This repository intentionally does not provide a full training pipeline.
    The factory focuses on instantiating architectures for inference/evaluation.
    """
    model_type = cfg.get("model", {}).get("type", cfg.get("model", ""))
    model_type = (model_type or "").lower()

    if model_type in ("tcn", "tcn_baseline"):
        from ppg_bio.models.tcn_baseline import build_embedding_classifier
        return build_embedding_classifier(**cfg.get("model", {}))

    if model_type in ("cgan", "pbgan", "gan"):
        from ppg_bio.models.cgan_pbgan import build_generator, build_discriminator
        # return builders; weight loading is user-provided
        return {"G": build_generator(**cfg.get("generator", {})),
                "D": build_discriminator(**cfg.get("discriminator", {}))}

    if model_type in ("cae", "conditional_ae"):
        from ppg_bio.models_torch.cae_film import build_conditional_cnn_ae_film_gated
        return build_conditional_cnn_ae_film_gated(**cfg.get("model", {}))

    if model_type in ("diffusion", "cond_diffusion"):
        from ppg_bio.models_torch.diffusion_cond import UNet1D_IDCond
        return UNet1D_IDCond(**cfg.get("model", {}))

    raise ValueError(f"Unknown model type: {model_type}")

def load_weights(model, weight_path: str):
    """Load weights for TF or Torch models if provided."""
    if weight_path is None or weight_path == "":
        return model

    if hasattr(model, "load_state_dict"):
        import torch
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        return model

    # Keras model
    try:
        model.load_weights(weight_path)
        return model
    except Exception:
        return model
