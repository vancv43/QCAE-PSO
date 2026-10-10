"""Exact McNemar test on paired downstream predictions of QCAE-PSO vs QCAE-GWO at sigma = 0.5.
Re-runs the published classifier set-up of classification_eval.py on the published checkpoints."""
import json, sys, os
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
os.chdir(HERE)
from models import QCAE
from data_utils import load_binary_mnist, add_gaussian_noise
R = json.load(open(os.path.join(CODE, "results", "results.json")))
xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
clf = LogisticRegression(max_iter=2000).fit(xtr.reshape(len(xtr), -1), ytr)
nte = torch.tensor(add_gaussian_noise(xte, 0.5, seed=2050)).unsqueeze(1)
pred = {}
for t in ["QCAE-PSO", "QCAE-GWO"]:
    hp = R["final"][t]["hyperparameters"]
    m = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    m.load_state_dict(torch.load(os.path.join(CODE, "results", "checkpoints", f"model_{t}.pt"))); m.eval()
    with torch.no_grad():
        pred[t] = clf.predict(m(nte).numpy()[:, 0].reshape(len(yte), -1))
a, b = pred["QCAE-PSO"] == yte, pred["QCAE-GWO"] == yte
n_pso_only, n_gwo_only = int((a & ~b).sum()), int((~a & b).sum())
p = binomtest(n_pso_only, n_pso_only + n_gwo_only, 0.5).pvalue
out = {"acc_pso": float(a.mean()), "acc_gwo": float(b.mean()), "pso_right_gwo_wrong": n_pso_only,
       "gwo_right_pso_wrong": n_gwo_only, "exact_mcnemar_p": float(p)}
json.dump(out, open("results/E4_mcnemar_pso_gwo.json", "w"), indent=2); print(out)
