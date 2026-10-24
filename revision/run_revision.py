# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""Robustness experiments of the paper.

  python run_revision.py retrain_seeds              # E1a
  python run_revision.py search --s 1 --space orig  # E1b (s in 1,2,3)
  python run_revision.py search --s 0 --space wide  # E2
  python run_revision.py diagnose                   # E3
"""
import argparse
import json
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(HERE)
import numpy as np
import torch

import metaheuristics as mh
from data_utils import load_binary_mnist, add_gaussian_noise
from models import ClassicalCAE, QCAE
from skimage.metrics import structural_similarity as ssim_fn

SEARCH_EPOCHS = 25
FINAL_EPOCHS = 80
N_POP, N_ITER, N_TRIALS = 6, 6, 2
SEARCH_SIGMA = 0.5
EVAL_SIGMAS = [0.25, 0.5, 0.75, 1.0]
NAME_OFFSET = {"PSO": 1, "GWO": 2, "HHO": 3, "BAT": 4, "E-HHO": 5}
OPTIMIZERS = {"PSO": mh.run_pso, "GWO": mh.run_gwo, "HHO": mh.run_hho,
              "BAT": mh.run_bat, "E-HHO": mh.run_ehho}
ORIG_RESULTS = os.path.join(ROOT, "results", "results.json")


def decode_wide(vec):
    vec = np.clip(vec, 0.0, 1.0)
    lr = 10 ** (-4 + vec[0] * 3)          # [1e-4, 1e-1]
    wd = 10 ** (-6 + vec[1] * 4)          # [1e-6, 1e-2], unchanged
    p_choices, hc_choices = [1, 2, 3], [4, 8, 16, 32, 64]
    p_idx = min(int(vec[2] * 3), 2)
    hc_idx = min(int(vec[3] * 5), 4)
    return {"lr": float(lr), "weight_decay": float(wd),
            "p_layers": p_choices[p_idx], "hidden_channels": hc_choices[hc_idx]}


def data():
    xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50, seed=0)
    return xtr[:300], xtr[300:], xte


def mean_ssim(recon_np, clean_np):
    return float(np.mean([ssim_fn(clean_np[i, 0], recon_np[i, 0], data_range=1.0)
                          for i in range(recon_np.shape[0])]))


def retrain_and_eval(kind, hp, x_trainval, x_test, seed):
    xtv = torch.tensor(x_trainval).unsqueeze(1)
    xte_t = torch.tensor(x_test).unsqueeze(1)
    ntv = torch.tensor(add_gaussian_noise(x_trainval, SEARCH_SIGMA, seed=33)).unsqueeze(1)
    torch.manual_seed(seed)
    if kind == "ccae":
        model = ClassicalCAE(hidden_channels=hp["hidden_channels"])
    else:
        model = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0))
    for _ in range(FINAL_EPOCHS):
        opt.zero_grad()
        loss = torch.mean((model(ntv) - xtv) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    per = {}
    for sigma in EVAL_SIGMAS:
        nte = torch.tensor(add_gaussian_noise(x_test, sigma, seed=2000 + int(sigma * 100))).unsqueeze(1)
        with torch.no_grad():
            per[str(sigma)] = mean_ssim(model(nte).numpy(), xte_t.numpy())
    return {"hyperparameters": hp, "final_train_loss": float(loss.item()),
            "ssim_by_sigma": per, "mean_ssim": float(np.mean(list(per.values())))}, model


def published_configs():
    orig = json.load(open(ORIG_RESULTS))
    cfg = {}
    for tag, r in orig["final"].items():
        cfg[tag] = ("ccae" if tag.startswith("CCAE") else "qcae", r["hyperparameters"])
    return cfg, orig


def cmd_retrain_seeds():
    x_train, x_val, x_test = data()
    x_trainval = np.concatenate([x_train, x_val], axis=0)
    cfg, orig = published_configs()
    out = {}
    for seed in [7, 11, 23, 37, 53]:
        for tag, (kind, hp) in cfg.items():
            t0 = time.time()
            res, _ = retrain_and_eval(kind, hp, x_trainval, x_test, seed)
            out.setdefault(tag, {})[str(seed)] = res
            print(f"[seed {seed}] {tag}: mean SSIM={res['mean_ssim']:.4f} "
                  f"(published {orig['final'][tag]['mean_ssim']:.4f}) {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("results/E1a_retrain_seeds.json", "w"), indent=2)


def cmd_search(s, space):
    if space == "wide":
        mh.decode = decode_wide  # make_fitness_fn resolves decode at call time
    decode = mh.decode
    x_train, x_val, x_test = data()
    nt = torch.tensor(add_gaussian_noise(x_train, SEARCH_SIGMA, seed=11)).unsqueeze(1)
    nv = torch.tensor(add_gaussian_noise(x_val, SEARCH_SIGMA, seed=12)).unsqueeze(1)
    xt = torch.tensor(x_train).unsqueeze(1)
    xv = torch.tensor(x_val).unsqueeze(1)
    base_fitness = mh.make_fitness_fn(nt, xt, nv, xv, epochs=SEARCH_EPOCHS, seed=s)

    search = {}
    for name, fn in OPTIMIZERS.items():
        trial_bests, hists, costs, evals = [], [], [], []
        for trial in range(N_TRIALS):
            log = []

            def fitness(vec, _log=log):
                f = base_fitness(vec)
                _log.append({"hp": decode(vec), "vec": [float(v) for v in np.clip(vec, 0, 1)], "f": float(f)})
                return f

            t0 = time.time()
            best_vec, best_f, hist = fn(fitness, n_pop=N_POP, n_iter=N_ITER,
                                        seed=1000 * s + 100 * trial + NAME_OFFSET[name])
            costs.append(time.time() - t0)
            trial_bests.append((best_vec, best_f))
            hists.append([float(h) for h in hist])
            evals.append(log)
            print(f"[s={s} {space}] [{name}] trial {trial}: best_f={best_f:.4f} "
                  f"decoded={decode(best_vec)} cost={costs[-1]:.1f}s", flush=True)
        bv, bf = min(trial_bests, key=lambda p: p[1])
        search[name] = {"best_vec": bv.tolist(), "best_hp": decode(bv), "best_fitness": float(bf),
                        "trial_fitnesses": [float(f) for _, f in trial_bests],
                        "histories": hists, "search_cost_seconds": costs, "evaluations": evals}

    x_trainval = np.concatenate([x_train, x_val], axis=0)
    final = {}
    for name, r in search.items():
        res, _ = retrain_and_eval("qcae", r["best_hp"], x_trainval, x_test, seed=7)
        final[f"QCAE-{name}"] = res
        print(f"[s={s} {space}] QCAE-{name}: {res['ssim_by_sigma']} mean={res['mean_ssim']:.4f}", flush=True)
    json.dump({"s": s, "space": space, "search": search, "final": final},
              open(f"results/E_search_s{s}_{space}.json", "w"), indent=2)


def cmd_diagnose():
    x_train, x_val, x_test = data()
    cfg, _ = published_configs()
    xte_t = torch.tensor(x_test).unsqueeze(1)
    nte = torch.tensor(add_gaussian_noise(x_test, 0.5, seed=2050)).unsqueeze(1)
    out = {}
    for tag, (kind, hp) in cfg.items():
        if kind == "ccae":
            model = ClassicalCAE(hidden_channels=hp["hidden_channels"])
        else:
            model = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
        model.load_state_dict(torch.load(os.path.join(ROOT, "results", "checkpoints", f"model_{tag}.pt")))
        model.eval()
        with torch.no_grad():
            recon = model(nte)
            d = {"recon_pixel_std_across_inputs_mean": float(recon.std(dim=0).mean()),
                 "recon_mean_intensity": float(recon.mean()),
                 "clean_pixel_std_across_inputs_mean": float(xte_t.std(dim=0).mean())}
            if kind == "qcae":
                y = model.pre_angle_norm(model.encoder(nte))
                t = torch.tanh(y)
                z = model.quantum(t * torch.pi)
                d["frac_saturated_tanh_gt_0.99"] = float((t.abs() > 0.99).float().mean())
                d["latent_z_std_across_inputs"] = [float(v) for v in z.std(dim=0)]
                d["angle_std_across_inputs"] = [float(v) for v in (t * torch.pi).std(dim=0)]
        out[tag] = d
        print(tag, json.dumps(d))
    json.dump(out, open("results/E3_diagnose.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--s", type=int, default=0)
    ap.add_argument("--space", default="orig")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    {"retrain_seeds": cmd_retrain_seeds, "diagnose": cmd_diagnose}.get(
        a.cmd, lambda: cmd_search(a.s, a.space))()
    print("DONE", flush=True)
