"""Draw Figures 3-7 of the paper as vector PDF files in figures/.

Data sources are the published result files only; nothing is re-run except a forward pass of
the published checkpoints for the qualitative reconstructions in Figure 5.
  results.json                 -> Fig. 3 (search convergence), Fig. 4 (SSIM by sigma, training loss)
  classification_results.json  -> Fig. 6 (confusion matrices), Fig. 7 (ROC, precision-recall)
  checkpoints/model_*.pt       -> Fig. 5 (reconstructions of the first three test images)
"""
import json
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures")
sys.path.insert(0, ROOT)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})
MM = 1 / 25.4

# One palette for every figure: manual baselines in greys, comparison optimizers in muted
# Okabe-Ito hues, the proposed PSO configuration in a saturated dark blue drawn last and thicker.
STYLE = {
    "Noisy (no denoising)": dict(color="#000000", ls=(0, (1.2, 1.4)), lw=1.0, marker=None),
    "CCAE-manual": dict(color="#7F7F7F", ls=(0, (4, 1.6)), lw=1.0, marker="s"),
    "QCAE-manual": dict(color="#BDBDBD", ls=(0, (1.5, 1.2)), lw=1.0, marker="D"),
    "QCAE-GWO": dict(color="#E69F00", ls="-", lw=1.0, marker="^"),
    "QCAE-HHO": dict(color="#56B4E9", ls="-", lw=1.0, marker="v"),
    "QCAE-BAT": dict(color="#CC79A7", ls="-", lw=1.0, marker="P"),
    "QCAE-E-HHO": dict(color="#009E73", ls="-", lw=1.0, marker="X"),
    "QCAE-PSO": dict(color="#08306B", ls="-", lw=1.9, marker="o"),
}
ORDER = ["CCAE-manual", "QCAE-manual", "QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO", "QCAE-PSO"]
OPT = ["GWO", "HHO", "BAT", "E-HHO", "PSO"]


def label(tag):
    return tag.replace("QCAE-PSO", "QCAE-PSO (ours)")


def panel_letter(ax, letter, fig, dx=-9, dy=2):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


R = json.load(open(os.path.join(ROOT, "results", "results.json")))
C = json.load(open(os.path.join(ROOT, "results", "classification_results.json")))["results"]
os.makedirs(OUT, exist_ok=True)

# ---------------- Fig. 3: search convergence ----------------
fig, ax = plt.subplots(figsize=(110 * MM, 62 * MM), layout="constrained")
for o in OPT:
    s = STYLE["QCAE-" + o]
    hs = np.array(R["search"][o]["histories"])
    h = hs.mean(axis=0)
    ax.fill_between(range(len(h)), hs.min(axis=0), hs.max(axis=0), color=s["color"], alpha=0.07, lw=0)
    ax.plot(range(len(h)), h, color=s["color"], lw=s["lw"], marker=s["marker"], ms=3.6 if o == "PSO" else 3,
            label=o + (" (ours)" if o == "PSO" else ""), zorder=3 if o == "PSO" else 2)
ax.set_xlabel("Search iteration")
ax.set_ylabel("Best fitness, 1 − validation SSIM")
ax.set_xticks(range(len(h)))
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, handlelength=2.2)
save(fig, "Fig3_search_convergence")

# ---------------- Fig. 4: SSIM by sigma + training loss ----------------
sig = ["0.25", "0.5", "0.75", "1.0"]
fig = plt.figure(figsize=(160 * MM, 66 * MM))
gs = GridSpec(2, 2, figure=fig, height_ratios=[0.22, 1], hspace=0.08, wspace=0.28,
              left=0.07, right=0.99, bottom=0.15, top=0.97)
axa, axb = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
x = np.arange(len(sig))
wbar = 0.115
for i, t in enumerate(ORDER):
    s = STYLE[t]
    v = [R["final"][t]["ssim_by_sigma"][k] for k in sig]
    axa.bar(x + (i - 3) * wbar, v, width=wbar * 0.92, color=s["color"],
            edgecolor="#08306B" if t == "QCAE-PSO" else "none", linewidth=0.6, label=label(t))
axa.set_xticks(x, ["0.25", "0.50", "0.75", "1.00"])
axa.set_xlabel("Gaussian noise σ")
axa.set_ylabel("Test-set SSIM")
axa.set_ylim(0, 0.35)
for t in ORDER:
    s = STYLE[t]
    axb.plot(np.arange(1, len(R["training_curves"][t]) + 1), R["training_curves"][t], color=s["color"],
             ls=s["ls"], lw=s["lw"], zorder=3 if t == "QCAE-PSO" else 2)
axb.set_xlabel("Epoch")
axb.set_ylabel("Training loss, MSE")
axb.set_xlim(0, 81)
axb.set_ylim(bottom=0)
lax = fig.add_subplot(gs[0, :]); lax.axis("off")
handles = [mpl.lines.Line2D([], [], color=STYLE[t]["color"], ls=STYLE[t]["ls"], lw=STYLE[t]["lw"] + 0.6,
                            label=label(t)) for t in ORDER]
lax.legend(handles=handles, loc="center", ncol=4, handlelength=2.4, columnspacing=1.6, borderaxespad=0)
panel_letter(axa, "a", fig, dx=-26)
panel_letter(axb, "b", fig, dx=-26)
save(fig, "Fig4_ssim_and_training")

# ---------------- Fig. 5: reconstructions ----------------
import torch  # noqa: E402
from models import ClassicalCAE, QCAE  # noqa: E402
from data_utils import load_binary_mnist, add_gaussian_noise  # noqa: E402

torch.set_num_threads(2)
xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
noisy = add_gaussian_noise(xte, 0.5, seed=2050)
idx = [0, 1, 2]  # first three held-out test images, as in the original figure
rec = {}
for t in ORDER:
    hp = R["final"][t]["hyperparameters"]
    m = ClassicalCAE(hidden_channels=hp["hidden_channels"]) if t.startswith("CCAE") else \
        QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "checkpoints", f"model_{t}.pt")))
    m.eval()
    with torch.no_grad():
        rec[t] = m(torch.tensor(noisy[idx]).unsqueeze(1)).numpy()[:, 0]
left = ["Original", "Noisy", "CCAE-manual", "QCAE-manual", "QCAE-PSO"]
right = ["QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO"]
fig = plt.figure(figsize=(120 * MM, 82 * MM))
outer = GridSpec(1, 2, figure=fig, wspace=0.42, left=0.15, right=0.99, top=0.95, bottom=0.02)
for block, rows, letter in [(0, left, "a"), (1, right, "b")]:
    g = outer[block].subgridspec(5, 3, hspace=0.05, wspace=0.05)
    for r, name in enumerate(rows):
        for j in range(3):
            ax = fig.add_subplot(g[r, j])
            img = xte[idx][j] if name == "Original" else noisy[idx][j] if name == "Noisy" else rec[name][j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("#606060")
            if j == 0:
                ax.set_ylabel(label(name).replace(" (ours)", "\n(ours)"), fontsize=6.8, rotation=0,
                              ha="right", va="center", labelpad=3)
            if r == 0 and j == 0:
                panel_letter(ax, letter, fig, dx=-44 if block == 0 else -40, dy=1)
save(fig, "Fig5_denoising_examples")

# ---------------- Fig. 6: confusion matrices ----------------
tags6 = ["Noisy (no denoising)"] + ORDER
fig, axes = plt.subplots(2, 4, figsize=(160 * MM, 84 * MM))
fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.11, wspace=0.42, hspace=0.62)
for ax, t in zip(axes.flat, tags6):
    cm = np.array(C[t]["confusion_matrix"])
    ax.pcolormesh(np.arange(-0.5, 2), np.arange(-0.5, 2), cm, cmap="Blues", vmin=0, vmax=50,
                  edgecolors="white", linewidth=0.6)  # vector cells
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5); ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > 25 else "black")
    ax.set_xticks([0, 1], ["0", "1"]); ax.set_yticks([0, 1], ["0", "1"])
    ax.set_xlabel("Predicted", labelpad=1.5); ax.set_ylabel("True", labelpad=1.5)
    ax.set_title(f"{label(t)}\naccuracy {C[t]['accuracy']:.3f}", fontsize=6.8,
                 fontweight="bold" if t == "QCAE-PSO" else "normal", pad=3)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
save(fig, "Fig6_confusion_matrices")

# ---------------- Fig. 7: ROC and precision-recall ----------------
tags7 = ["Noisy (no denoising)"] + ORDER
fig, (axa, axb) = plt.subplots(1, 2, figsize=(160 * MM, 76 * MM))
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.25, wspace=0.26)
for t in tags7:
    s = STYLE[t]
    z = 3 if t == "QCAE-PSO" else 2
    axa.plot(C[t]["fpr"], C[t]["tpr"], color=s["color"], ls=s["ls"], lw=s["lw"], zorder=z,
             label=label(t))
    axb.plot(C[t]["pr_recall"], C[t]["pr_precision"], color=s["color"], ls=s["ls"], lw=s["lw"], zorder=z,
             label=label(t))
axa.plot([0, 1], [0, 1], color="#9E9E9E", lw=0.6, ls=":", zorder=1)
axb.axhline(0.5, color="#9E9E9E", lw=0.6, ls=":", zorder=1)
axa.set_xlabel("False positive rate"); axa.set_ylabel("True positive rate")
axb.set_xlabel("Recall"); axb.set_ylabel("Precision")
for ax in (axa, axb):
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.03); ax.set_aspect("auto")
h7, l7 = axa.get_legend_handles_labels()
fig.legend(h7, l7, loc="lower center", ncol=4, handlelength=2.4, columnspacing=1.6, bbox_to_anchor=(0.53, 0.01))
panel_letter(axa, "a", fig, dx=-26)
panel_letter(axb, "b", fig, dx=-26)
save(fig, "Fig7_roc_pr")
