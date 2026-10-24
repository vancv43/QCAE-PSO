# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""Re-evaluate the seven published checkpoints on the held-out test split.

Loads results/checkpoints/model_<tag>.pt, rebuilds the same test images and noise realizations as
run_search.py, and recomputes the test SSIM at every noise level. The values are compared with
results/results.json, which run_search.py wrote when it trained the models, so Table 2 of the paper can be
checked in a few seconds without retraining. Needs the MNIST files, see README.md.

Usage, from the repository root:
    python evaluate_checkpoints.py
"""
import json
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from data_utils import load_binary_mnist, add_gaussian_noise
from models import ClassicalCAE, QCAE

EVAL_SIGMAS = [0.25, 0.5, 0.75, 1.0]
TAGS = ["CCAE-manual", "QCAE-manual", "QCAE-GWO", "QCAE-HHO", "QCAE-BAT", "QCAE-E-HHO", "QCAE-PSO"]

with open("results/results.json") as f:
    R = json.load(f)

# same split as run_search.py: the 100 test images are never used for training or search
_, _, x_test, _ = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
clean = torch.tensor(x_test).unsqueeze(1)

worst = 0.0
print(f"{'model':<12} " + " ".join(f"sigma={s:<5}" for s in EVAL_SIGMAS) + "  mean    stored  |diff|")
for tag in TAGS:
    hp = R["final"][tag]["hyperparameters"]
    if tag == "CCAE-manual":
        model = ClassicalCAE(hidden_channels=hp["hidden_channels"])
    else:
        model = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    model.load_state_dict(torch.load(f"results/checkpoints/model_{tag}.pt"))
    model.eval()
    per_sigma = []
    for sigma in EVAL_SIGMAS:
        noisy = torch.tensor(add_gaussian_noise(x_test, sigma, seed=2000 + int(sigma * 100))).unsqueeze(1)
        with torch.no_grad():
            out = model(noisy).numpy()
        per_sigma.append(float(np.mean([ssim_fn(clean.numpy()[i, 0], out[i, 0], data_range=1.0)
                                        for i in range(out.shape[0])])))
    stored = [R["final"][tag]["ssim_by_sigma"][str(s)] for s in EVAL_SIGMAS]
    diff = max(abs(a - b) for a, b in zip(per_sigma, stored))
    worst = max(worst, diff)
    print(f"{tag:<12} " + " ".join(f"{v:<11.4f}" for v in per_sigma)
          + f"  {np.mean(per_sigma):.4f}  {R['final'][tag]['mean_ssim']:.4f}  {diff:.1e}")

print(f"largest difference from results/results.json: {worst:.1e}")
sys.exit(0 if worst < 1e-6 else 1)
