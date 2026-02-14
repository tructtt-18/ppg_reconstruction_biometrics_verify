from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

def reconstruction_error_l1(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x - x_hat), axis=tuple(range(1, x.ndim)))

def reconstruct(model: Any, x: Any, y_claim: Any = None) -> Any:
    """Unified reconstruction entrypoint.

    - For AE: model(x, y_claim) or model(x) depending on implementation.
    - For diffusion: user should call the notebook-adapted helpers in diffusion_cond.py.
    - For GAN: reconstruction/inversion is not bundled into a one-click pipeline.
    """
    # best-effort generic forward
    if hasattr(model, "__call__"):
        try:
            return model(x, y_claim)
        except TypeError:
            return model(x)
    raise TypeError("Unsupported model type for reconstruct()")
