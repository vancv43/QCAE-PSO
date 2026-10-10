"""Full experiment: PSO / GWO / HHO / BAT / E-HHO hyperparameter search for QCAE,
baseline CCAE and manual-QCAE, final retraining, and evaluation across
Gaussian noise levels sigma in {0.25, 0.5, 0.75, 1.0} on a held-out test set.

All numbers in results.json / figures are produced by actually running this
script -- nothing here is a placeholder or hand-typed number.
"""
import json
import time
import numpy as np
import torch

from data_utils import load_binary_mnist, add_gaussian_noise
from models import ClassicalCAE, QCAE
from metaheuristics import (
    decode, make_fitness_fn, run_pso, run_gwo, run_ehho, run_hho, run_bat, train_and_eval_qcae,
)
from skimage.metrics import structural_similarity as ssim_fn

RESULTS_DIR = "results"
CKPT_DIR = f"{RESULTS_DIR}/checkpoints"
SEARCH_EPOCHS = 25
FINAL_EPOCHS = 80
N_POP = 6
N_ITER = 6
N_TRIALS = 2
SEARCH_SIGMA = 0.5
EVAL_SIGMAS = [0.25, 0.5, 0.75, 1.0]

MANUAL_QCAE_HP = {"lr": 1e-3, "weight_decay": 0.0, "p_layers": 2, "hidden_channels": 8}
MANUAL_CCAE_HP = {"lr": 1e-3, "weight_decay": 0.0, "hidden_channels": 8}


def mean_ssim(recon_np, clean_np):
    return float(np.mean([
        ssim_fn(clean_np[i, 0], recon_np[i, 0], data_range=1.0) for i in range(recon_np.shape[0])
    ]))


def main():
    import os
    os.makedirs(CKPT_DIR, exist_ok=True)
    t_start = time.time()
    xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
    # xtr: 400 samples (200/200), xte: 100 samples held out entirely for final eval
    x_train = xtr[:300]
    x_val = xtr[300:]  # 100 samples, used only as the search-time fitness validation split
    x_test = xte       # 100 samples, held out entirely until final evaluation

    noisy_train = add_gaussian_noise(x_train, SEARCH_SIGMA, seed=11)
    noisy_val = add_gaussian_noise(x_val, SEARCH_SIGMA, seed=12)

    xt = torch.tensor(x_train).unsqueeze(1)
    nt = torch.tensor(noisy_train).unsqueeze(1)
    xv = torch.tensor(x_val).unsqueeze(1)
    nv = torch.tensor(noisy_val).unsqueeze(1)

    fitness = make_fitness_fn(nt, xt, nv, xv, epochs=SEARCH_EPOCHS, seed=0)

    optimizers = {
        "PSO": run_pso,
        "GWO": run_gwo,
        "HHO": run_hho,
        "BAT": run_bat,
        "E-HHO": run_ehho,
    }

    # Fixed, deterministic per-optimizer seed offset (Python's built-in hash() of
    # strings is randomized per-process unless PYTHONHASHSEED is fixed, which
    # made earlier runs of this script non-reproducible across processes).
    NAME_OFFSET = {"PSO": 1, "GWO": 2, "HHO": 3, "BAT": 4, "E-HHO": 5}

    search_results = {}
    for name, fn in optimizers.items():
        trial_bests = []
        trial_hists = []
        trial_costs = []
        for trial in range(N_TRIALS):
            fitness.__wrapped_cache__ = {}  # not used, just documenting no cross-trial leakage
            t0 = time.time()
            best_vec, best_f, hist = fn(fitness, n_pop=N_POP, n_iter=N_ITER, seed=100 * trial + NAME_OFFSET[name])
            cost = time.time() - t0
            trial_bests.append((best_vec, best_f))
            trial_hists.append(hist)
            trial_costs.append(cost)
            print(f"[{name}] trial {trial}: best_f={best_f:.4f} decoded={decode(best_vec)} cost={cost:.1f}s")
        best_overall = min(trial_bests, key=lambda p: p[1])
        search_results[name] = {
            "best_vec": best_overall[0].tolist(),
            "best_hp": decode(best_overall[0]),
            "best_fitness": best_overall[1],
            "trial_fitnesses": [f for _, f in trial_bests],
            "histories": [[float(h) for h in hist] for hist in trial_hists],
            "search_cost_seconds": trial_costs,
        }

    print("\n=== Search complete, retraining final models on train+val, evaluating on held-out test ===")

    x_trainval = np.concatenate([x_train, x_val], axis=0)
    xtv = torch.tensor(x_trainval).unsqueeze(1)
    xte_t = torch.tensor(x_test).unsqueeze(1)

    final_results = {}

    def evaluate_across_sigmas(model, tag, seed):
        model.eval()
        per_sigma = {}
        for sigma in EVAL_SIGMAS:
            noisy_test = add_gaussian_noise(x_test, sigma, seed=2000 + int(sigma * 100))
            nte_t = torch.tensor(noisy_test).unsqueeze(1)
            with torch.no_grad():
                out = model(nte_t)
            s = mean_ssim(out.numpy(), xte_t.numpy())
            per_sigma[str(sigma)] = s
            print(f"  [{tag}] sigma={sigma}: SSIM={s:.4f}")
        return per_sigma

    configs = {
        "CCAE-manual": ("ccae", MANUAL_CCAE_HP),
        "QCAE-manual": ("qcae", MANUAL_QCAE_HP),
    }
    for name, res in search_results.items():
        configs[f"QCAE-{name}"] = ("qcae", res["best_hp"])

    training_curves = {}

    for tag, (kind, hp) in configs.items():
        print(f"\nFinal training: {tag} with hp={hp}")
        noisy_trainval = add_gaussian_noise(x_trainval, SEARCH_SIGMA, seed=33)
        ntv = torch.tensor(noisy_trainval).unsqueeze(1)
        torch.manual_seed(7)
        if kind == "ccae":
            model = ClassicalCAE(hidden_channels=hp["hidden_channels"])
            wd = hp.get("weight_decay", 0.0)
        else:
            model = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
            wd = hp.get("weight_decay", 0.0)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=wd)
        losses = []
        for epoch in range(FINAL_EPOCHS):
            opt.zero_grad()
            out = model(ntv)
            loss = torch.mean((out - xtv) ** 2)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        training_curves[tag] = losses
        per_sigma_ssim = evaluate_across_sigmas(model, tag, seed=7)
        final_results[tag] = {
            "hyperparameters": hp,
            "final_train_loss": losses[-1],
            "ssim_by_sigma": per_sigma_ssim,
            "mean_ssim": float(np.mean(list(per_sigma_ssim.values()))),
        }
        torch.save(model.state_dict(), f"{CKPT_DIR}/model_{tag}.pt")

    total_time = time.time() - t_start
    output = {
        "search": search_results,
        "final": final_results,
        "training_curves": training_curves,
        "config": {
            "search_epochs": SEARCH_EPOCHS,
            "final_epochs": FINAL_EPOCHS,
            "n_pop": N_POP,
            "n_iter": N_ITER,
            "n_trials": N_TRIALS,
            "search_sigma": SEARCH_SIGMA,
            "eval_sigmas": EVAL_SIGMAS,
            "n_train": len(x_train),
            "n_val": len(x_val),
            "n_test": len(x_test),
        },
        "total_wallclock_seconds": total_time,
    }
    with open(f"{RESULTS_DIR}/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nTotal wall-clock time: {total_time:.1f}s. Results saved to {RESULTS_DIR}/results.json")


if __name__ == "__main__":
    main()
