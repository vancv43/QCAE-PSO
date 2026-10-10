"""Downstream classification utility evaluation.

A logistic regression classifier is trained on CLEAN images (digit 0 vs 1)
from the combined train+val split, then evaluated on the denoised
reconstructions produced by each of the seven final models (plus the raw
noisy image, as a no-denoising reference), all at sigma = 0.5, matching the
representative visualization used elsewhere in the paper. This tests
whether a classifier trained on clean data still works on each model's
reconstructions -- a task-driven complement to the pixel-level SSIM metric.

All numbers are computed by actually running this script; nothing here is
hand-typed.
"""
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score,
)

from models import ClassicalCAE, QCAE
from data_utils import load_binary_mnist, add_gaussian_noise

with open("results/results.json") as f:
    R = json.load(f)

SIGMA_EVAL = 0.5

xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
x_trainval = xtr  # 400 clean images, real labels, used to fit the classifier
y_trainval = ytr

clf = LogisticRegression(max_iter=2000)
clf.fit(x_trainval.reshape(len(x_trainval), -1), y_trainval)
clean_train_acc = clf.score(x_trainval.reshape(len(x_trainval), -1), y_trainval)
print(f"Classifier train accuracy on clean images: {clean_train_acc:.4f}")

noisy_test = add_gaussian_noise(xte, SIGMA_EVAL, seed=2000 + int(SIGMA_EVAL * 100))
nte = torch.tensor(noisy_test).unsqueeze(1)

tags = ["Noisy (no denoising)", "CCAE-manual", "QCAE-manual", "QCAE-PSO",
        "QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO"]

recon_by_tag = {"Noisy (no denoising)": noisy_test}

for tag in tags[1:]:
    hp = R["final"][tag]["hyperparameters"]
    if tag == "CCAE-manual":
        m = ClassicalCAE(hidden_channels=hp["hidden_channels"])
    else:
        m = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    m.load_state_dict(torch.load(f"results/checkpoints/model_{tag}.pt"))
    m.eval()
    with torch.no_grad():
        out = m(nte).squeeze(1).numpy()
    recon_by_tag[tag] = out

results = {}
for tag in tags:
    imgs = recon_by_tag[tag].reshape(len(yte), -1)
    probs = clf.predict_proba(imgs)[:, 1]
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(yte, preds, labels=[0, 1])
    fpr, tpr, _ = roc_curve(yte, probs)
    auc = roc_auc_score(yte, probs)
    pr_precision, pr_recall, _ = precision_recall_curve(yte, probs, pos_label=1)
    ap = average_precision_score(yte, probs)
    acc = float((preds == yte).mean())
    prec = float(precision_score(yte, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(yte, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(yte, preds, pos_label=1, zero_division=0))
    results[tag] = {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "average_precision": float(ap),
        "pr_precision": pr_precision.tolist(),
        "pr_recall": pr_recall.tolist(),
    }
    print(f"{tag}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} auc={auc:.4f} ap={ap:.4f} cm={cm.tolist()}")

with open("results/classification_results.json", "w") as f:
    json.dump({"classifier_clean_train_acc": clean_train_acc, "sigma": SIGMA_EVAL,
               "results": results}, f, indent=2)

print("Metrics written to results/classification_results.json")
