import os
import random
from typing import Optional

import numpy as np

def set_seed(seed: int = 1337) -> None:
    """Set random seeds for reproducibility (NumPy + Python random)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_device(prefer: str = "cuda") -> str:
    """Return 'cuda' if available, else 'cpu'."""
    if prefer == "cuda":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"
