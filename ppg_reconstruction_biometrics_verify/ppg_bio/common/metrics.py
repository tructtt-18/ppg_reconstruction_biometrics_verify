from __future__ import annotations

import numpy as np

def compute_eer(fpr, tpr, thr):
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return 0.5*(fpr[idx] + fnr[idx]), thr[idx]

eer, thr_eer = compute_eer(fpr, tpr, thr)


def far_frr_at_threshold(scores, labels, tau):
    # FAR = P(accept|impostor), FRR = P(reject|genuine)
    imp = scores[labels==0]; gen = scores[labels==1]
    FAR = np.mean(imp >= tau) if len(imp)>0 else 0.0
    FRR = np.mean(gen <  tau) if len(gen)>0 else 0.0
    return FAR, FRR

FAR_eer, FRR_eer = far_frr_at_threshold(scores_all, labels_all, thr_eer)

print(f"Open-set ROC-AUC: {roc_auc:.4f}")
print(f"Open-set EER (global): {eer:.4f} @ tau={thr_eer:.4f}")
print(f"FAR/FRR @ global-EER tau: FAR={FAR_eer:.4f}, FRR={FRR_eer:.4f}")


emb_val = embed(X_val)
np.save(os.path.join(OUT_DIR, "emb_val_tcn.npy"), emb_val.astype(np.float32))

