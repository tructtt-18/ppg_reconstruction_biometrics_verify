from __future__ import annotations

import os
from typing import Any, Dict, Tuple, Optional

import numpy as np

def load_npz(path: str) -> Dict[str, Any]:
    """Load an .npz file and return a dict of arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}

def save_npz(path: str, **arrays: Any) -> None:
    """Save arrays to .npz."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=False)

def save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, arr)
