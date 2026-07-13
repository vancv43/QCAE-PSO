"""Generate all figures for the paper from results/results.json (real numbers only)."""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import ClassicalCAE, QCAE
from data_utils import load_binary_mnist, add_gaussian_noise

with open("results/results.json") as f:
    R = json.load(f)

FIG_DIR = "figures"

# Figure: search convergence curves (mean over 2 trials, best fitness vs iteration)
plt.figure(figsize=(5.5, 3.8))
colors = {"PSO": "tab:blue", "GWO": "tab:orange", "HHO": "tab:purple", "BAT": "tab:brown", "E-HHO": "tab:green"}
for name in ["PSO", "GWO", "HHO", "BAT", "E-HHO"]:
    hists = np.array(R["search"][name]["histories"])  # (n_trials, n_iter+1)
    mean_hist = hists.mean(axis=0)
    plt.plot(range(len(mean_hist)), mean_hist, marker="o", label=name, color=colors[name])
plt.xlabel("Iteration")
plt.ylabel("Best fitness (1 - validation SSIM)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/search_convergence.png", dpi=300)
plt.close()

# Figure: final training loss curves
plt.figure(figsize=(5.5, 3.8))
style = {
    "CCAE-manual": ("tab:gray", "--"),
    "QCAE-manual": ("tab:red", "--"),
    "QCAE-PSO": ("tab:blue", "-"),
    "QCAE-GWO": ("tab:orange", "-"),
    "QCAE-HHO": ("tab:purple", "-"),
    "QCAE-BAT": ("tab:brown", "-"),
    "QCAE-E-HHO": ("tab:green", "-"),
}
for tag, losses in R["training_curves"].items():
    c, ls = style[tag]
    plt.plot(losses, label=tag, color=c, linestyle=ls, linewidth=1.4)
plt.xlabel("Epoch")
plt.ylabel("Training loss (MSE)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/training_curves.png", dpi=300)
plt.close()

# Figure: SSIM by sigma bar chart
sigmas = R["config"]["eval_sigmas"]
tags = ["CCAE-manual", "QCAE-manual", "QCAE-PSO", "QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO"]
x = np.arange(len(sigmas))
width = 0.11
plt.figure(figsize=(7.5, 5.18))
for i, tag in enumerate(tags):
    vals = [R["final"][tag]["ssim_by_sigma"][str(s)] for s in sigmas]
    c, _ = style[tag]
    plt.bar(x + i * width, vals, width=width, label=tag, color=c)
plt.xticks(x + width * 3, [str(s) for s in sigmas])
plt.xlabel("Gaussian noise sigma")
plt.ylabel("Test-set SSIM")
plt.legend(fontsize=7, ncol=3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/ssim_by_sigma.png", dpi=300)
plt.close()

# Figure: qualitative denoising examples at sigma=0.5
xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
sigma_show = 0.5
noisy_test = add_gaussian_noise(xte, sigma_show, seed=2000 + int(sigma_show * 100))
idx = [0, 1, 2]  # first three held-out test images (kept small: now 7 model rows)

models = {}
for tag in tags:
    hp = R["final"][tag]["hyperparameters"]
    if tag == "CCAE-manual":
        m = ClassicalCAE(hidden_channels=hp["hidden_channels"])
    else:
        m = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    m.load_state_dict(torch.load(f"results/model_{tag}.pt"))
    m.eval()
    models[tag] = m

nte = torch.tensor(noisy_test[idx]).unsqueeze(1)
xte_show = xte[idx]

recon_by_tag = {}
for tag in tags:
    with torch.no_grad():
        recon_by_tag[tag] = models[tag](nte).numpy()

# Split into two halves (5 rows / 4 rows) so the paper can place them as a
# side-by-side subfigure pair instead of one overly tall single image.
row_groups = {
    "a": ["Original", "Noisy", "CCAE-manual", "QCAE-manual", "QCAE-PSO"],
    "b": ["QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO"],
}

for part, rows in row_groups.items():
    fig, axes = plt.subplots(len(rows), len(idx), figsize=(1.15 * len(idx), 0.85 * len(rows)))
    plt.subplots_adjust(hspace=0.08, wspace=0.08)
    for j in range(len(idx)):
        for r_i, name in enumerate(rows):
            if name == "Original":
                img = xte_show[j]
            elif name == "Noisy":
                img = noisy_test[idx][j]
            else:
                img = recon_by_tag[name][j, 0]
            axes[r_i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
    for r_i, name in enumerate(rows):
        axes[r_i, 0].set_ylabel(name, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/denoising_examples_{part}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Figures written to", FIG_DIR)
