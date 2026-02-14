#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np

from ppg_bio.common.io import load_npz
from ppg_bio.common.metrics import compute_eer, far_frr_at_threshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", required=True, help="npz with key: scores (or RE) and labels")
    ap.add_argument("--scores_key", default="scores")
    ap.add_argument("--labels_key", default="labels")
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()

    d = load_npz(args.input_npz)
    scores = d.get(args.scores_key, None)
    if scores is None:
        scores = d.get("RE", None)
    labels = d.get(args.labels_key, None)

    if scores is None or labels is None:
        raise KeyError("Need scores (or RE) and labels in input npz.")

    eer, tau_eer = compute_eer(scores, labels)
    print(f"EER: {eer:.6f}, tau@EER: {tau_eer:.6f}")

    if args.tau is not None:
        far, frr = far_frr_at_threshold(scores, labels, args.tau)
        print(f"tau={args.tau:.6f} FAR={far:.6f} FRR={frr:.6f}")

if __name__ == "__main__":
    main()
