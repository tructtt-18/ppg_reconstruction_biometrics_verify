#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from ppg_bio.common.io import load_npz, save_npz
from ppg_bio.api.build_model import load_config, build_model_from_config, load_weights
from ppg_bio.api.reconstruct import reconstruct, reconstruction_error_l1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--input_npz", required=True, help="npz with keys: X, y (optional)")
    ap.add_argument("--output_npz", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model_from_config(cfg)
    model = load_weights(model, args.weights)

    data = load_npz(args.input_npz)
    X = data.get("X", data.get("test_X", None))
    if X is None:
        raise KeyError("Input npz must contain key 'X' or 'test_X'.")

    # Note: this is a lightweight inference example.
    # For diffusion/cGAN, use the dedicated helpers in their modules.
    X_hat = reconstruct(model, X)
    re_l1 = reconstruction_error_l1(X, X_hat)

    save_npz(args.output_npz, X_hat=X_hat, RE=re_l1)

if __name__ == "__main__":
    main()
