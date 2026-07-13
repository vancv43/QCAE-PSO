"""PSO, GWO, and E-HHO search over QCAE hyperparameters.

Search space (4-D, continuous vector in [0,1]^4 decoded to):
  0: learning rate, log-scale in [1e-4, 1e-2]
  1: weight decay,  log-scale in [1e-6, 1e-2]
  2: QAOA depth p,  discrete in {1, 2, 3}
  3: hidden channels, discrete in {4, 8, 16}

Fitness = 1 - mean SSIM on the held-out validation split (lower is better),
matching the "lower fitness = better" convention used throughout.

E-HHO reuses the three enhancement mechanisms from the author's prior HHO
hybrid (chaotic logistic-map initialization, oscillating escaping-energy
schedule, elite opposition-based learning); PSO and GWO follow the standard
formulations (Kennedy and Eberhart, 1995; Mirjalili et al., 2014) and HHO
follows Heidari et al. (2019).
"""
import numpy as np
import torch
import time
from skimage.metrics import structural_similarity as ssim_fn

from models import ClassicalCAE, QCAE
from data_utils import add_gaussian_noise

DIM = 4
BOUNDS_LOW = np.array([0.0, 0.0, 0.0, 0.0])
BOUNDS_HIGH = np.array([1.0, 1.0, 1.0, 1.0])

P_CHOICES = [1, 2, 3]
HC_CHOICES = [4, 8, 16]


def decode(vec):
    vec = np.clip(vec, 0.0, 1.0)
    lr = 10 ** (np.log10(1e-4) + vec[0] * (np.log10(1e-2) - np.log10(1e-4)))
    wd = 10 ** (np.log10(1e-6) + vec[1] * (np.log10(1e-2) - np.log10(1e-6)))
    p_idx = min(int(vec[2] * len(P_CHOICES)), len(P_CHOICES) - 1)
    hc_idx = min(int(vec[3] * len(HC_CHOICES)), len(HC_CHOICES) - 1)
    return {
        "lr": float(lr),
        "weight_decay": float(wd),
        "p_layers": P_CHOICES[p_idx],
        "hidden_channels": HC_CHOICES[hc_idx],
    }


def _mean_ssim(recon, clean):
    vals = []
    recon_np = recon.detach().numpy()
    clean_np = clean.numpy()
    for i in range(recon_np.shape[0]):
        vals.append(ssim_fn(clean_np[i, 0], recon_np[i, 0], data_range=1.0))
    return float(np.mean(vals))


def train_and_eval_qcae(hp, x_train_noisy, x_train_clean, x_val_noisy, x_val_clean,
                         epochs=15, seed=0, sigma_train=None):
    torch.manual_seed(seed)
    model = QCAE(p_layers=hp["p_layers"], hidden_channels=hp["hidden_channels"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    for _ in range(epochs):
        opt.zero_grad()
        out = model(x_train_noisy)
        loss = torch.mean((out - x_train_clean) ** 2)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_out = model(x_val_noisy)
        val_ssim = _mean_ssim(val_out, x_val_clean)
    return model, val_ssim, loss.item()


def make_fitness_fn(x_train_noisy, x_train_clean, x_val_noisy, x_val_clean, epochs=15, seed=0):
    cache = {}

    def fitness(vec):
        key = tuple(np.round(vec, 6))
        if key in cache:
            return cache[key]
        hp = decode(vec)
        _, val_ssim, _ = train_and_eval_qcae(
            hp, x_train_noisy, x_train_clean, x_val_noisy, x_val_clean, epochs=epochs, seed=seed
        )
        f = 1.0 - val_ssim
        cache[key] = f
        return f

    return fitness


def _clip(x):
    return np.clip(x, BOUNDS_LOW, BOUNDS_HIGH)


def run_pso(fitness, n_pop=6, n_iter=6, seed=0, w=0.6, c1=1.5, c2=1.5):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_pop, DIM))
    V = np.zeros((n_pop, DIM))
    pbest = X.copy()
    pbest_f = np.array([fitness(x) for x in X])
    gbest_idx = np.argmin(pbest_f)
    gbest = pbest[gbest_idx].copy()
    gbest_f = pbest_f[gbest_idx]
    history = [gbest_f]
    for t in range(n_iter):
        r1 = rng.uniform(0, 1, size=(n_pop, DIM))
        r2 = rng.uniform(0, 1, size=(n_pop, DIM))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = _clip(X + V)
        f = np.array([fitness(x) for x in X])
        improved = f < pbest_f
        pbest[improved] = X[improved]
        pbest_f[improved] = f[improved]
        idx = np.argmin(pbest_f)
        if pbest_f[idx] < gbest_f:
            gbest_f = pbest_f[idx]
            gbest = pbest[idx].copy()
        history.append(gbest_f)
    return gbest, gbest_f, history


def run_gwo(fitness, n_pop=6, n_iter=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, size=(n_pop, DIM))
    f = np.array([fitness(x) for x in X])
    order = np.argsort(f)
    alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
    f_alpha = f[order[0]]
    history = [f_alpha]
    for t in range(n_iter):
        a = 2 - 2 * t / max(n_iter - 1, 1)
        for i in range(n_pop):
            new_pos = np.zeros(DIM)
            for leader in (alpha, beta, delta):
                r1 = rng.uniform(0, 1, size=DIM)
                r2 = rng.uniform(0, 1, size=DIM)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - X[i])
                new_pos += leader - A * D
            X[i] = _clip(new_pos / 3.0)
        f = np.array([fitness(x) for x in X])
        order = np.argsort(f)
        candidates = np.vstack([alpha, beta, delta, X])
        cand_f = np.concatenate([[f_alpha, fitness(beta), fitness(delta)], f])
        best_order = np.argsort(cand_f)
        alpha, beta, delta = candidates[best_order[0]].copy(), candidates[best_order[1]].copy(), candidates[best_order[2]].copy()
        f_alpha = cand_f[best_order[0]]
        history.append(f_alpha)
    return alpha, f_alpha, history


def run_ehho(fitness, n_pop=6, n_iter=6, seed=0, alpha_osc=0.3):
    """Enhanced HHO: chaotic logistic-map init, oscillating escaping energy,
    elite opposition-based learning (mechanisms from the author's prior work)."""
    rng = np.random.RandomState(seed)

    # chaotic initialization via logistic map
    z = rng.uniform(0.05, 0.95)
    X = np.zeros((n_pop, DIM))
    for i in range(n_pop):
        for d in range(DIM):
            z = 4 * z * (1 - z)
            X[i, d] = z
    X = _clip(X)

    f = np.array([fitness(x) for x in X])
    best_idx = np.argmin(f)
    rabbit = X[best_idx].copy()
    f_rabbit = f[best_idx]
    history = [f_rabbit]

    T = n_iter
    for t in range(1, T + 1):
        E0 = rng.uniform(-1, 1)
        E = 2 * E0 * (1 - t / T) * (1 + alpha_osc * np.sin(2 * np.pi * t / T))
        for i in range(n_pop):
            if abs(E) >= 1:
                if rng.uniform() < 0.5:
                    rand_hawk = X[rng.randint(n_pop)]
                    X[i] = rand_hawk - rng.uniform() * np.abs(rand_hawk - 2 * rng.uniform() * X[i])
                else:
                    mean_pos = X.mean(axis=0)
                    X[i] = (rabbit - mean_pos) - rng.uniform() * (BOUNDS_LOW + rng.uniform() * (BOUNDS_HIGH - BOUNDS_LOW))
            else:
                jump = 2 * (1 - rng.uniform())
                X[i] = rabbit - E * np.abs(jump * rabbit - X[i])
            X[i] = _clip(X[i])
        f = np.array([fitness(x) for x in X])
        idx = np.argmin(f)
        if f[idx] < f_rabbit:
            f_rabbit = f[idx]
            rabbit = X[idx].copy()

        # elite opposition-based learning
        x_opp = _clip(BOUNDS_LOW + BOUNDS_HIGH - rabbit)
        f_opp = fitness(x_opp)
        if f_opp < f_rabbit:
            f_rabbit = f_opp
            rabbit = x_opp

        history.append(f_rabbit)
    return rabbit, f_rabbit, history


def run_hho(fitness, n_pop=6, n_iter=6, seed=0):
    """Plain Harris Hawks Optimization (Heidari et al., 2019): uniform random
    initialization and a linearly decaying escaping energy, with no chaotic
    initialization, no oscillating-energy schedule, and no elite
    opposition-based learning. Included as a baseline to isolate the
    contribution of the three E-HHO enhancement mechanisms."""
    rng = np.random.RandomState(seed)

    X = rng.uniform(0, 1, size=(n_pop, DIM))
    f = np.array([fitness(x) for x in X])
    best_idx = np.argmin(f)
    rabbit = X[best_idx].copy()
    f_rabbit = f[best_idx]
    history = [f_rabbit]

    T = n_iter
    for t in range(1, T + 1):
        E0 = rng.uniform(-1, 1)
        E = 2 * E0 * (1 - t / T)
        for i in range(n_pop):
            if abs(E) >= 1:
                if rng.uniform() < 0.5:
                    rand_hawk = X[rng.randint(n_pop)]
                    X[i] = rand_hawk - rng.uniform() * np.abs(rand_hawk - 2 * rng.uniform() * X[i])
                else:
                    mean_pos = X.mean(axis=0)
                    X[i] = (rabbit - mean_pos) - rng.uniform() * (BOUNDS_LOW + rng.uniform() * (BOUNDS_HIGH - BOUNDS_LOW))
            else:
                jump = 2 * (1 - rng.uniform())
                X[i] = rabbit - E * np.abs(jump * rabbit - X[i])
            X[i] = _clip(X[i])
        f = np.array([fitness(x) for x in X])
        idx = np.argmin(f)
        if f[idx] < f_rabbit:
            f_rabbit = f[idx]
            rabbit = X[idx].copy()
        history.append(f_rabbit)
    return rabbit, f_rabbit, history


def run_bat(fitness, n_pop=6, n_iter=6, seed=0, f_min=0.0, f_max=2.0,
            alpha=0.9, gamma=0.9, a0=1.0, r0=0.5):
    """Bat Algorithm (Yang, 2010): frequency-tuned velocity updates towards
    the current best, with loudness-gated acceptance of new solutions and a
    pulse-rate-gated local random walk around the best solution."""
    rng = np.random.RandomState(seed)

    X = rng.uniform(0, 1, size=(n_pop, DIM))
    V = np.zeros((n_pop, DIM))
    A = np.full(n_pop, a0)
    r = np.full(n_pop, r0)
    f_val = np.array([fitness(x) for x in X])
    best_idx = np.argmin(f_val)
    best = X[best_idx].copy()
    f_best = f_val[best_idx]
    history = [f_best]

    for t in range(1, n_iter + 1):
        for i in range(n_pop):
            freq = f_min + (f_max - f_min) * rng.uniform()
            V[i] = V[i] + (X[i] - best) * freq
            candidate = _clip(X[i] + V[i])
            if rng.uniform() > r[i]:
                candidate = _clip(best + 0.01 * rng.normal(size=DIM) * A.mean())
            f_candidate = fitness(candidate)
            if f_candidate <= f_val[i] and rng.uniform() < A[i]:
                X[i] = candidate
                f_val[i] = f_candidate
                A[i] *= alpha
                r[i] = r0 * (1 - np.exp(-gamma * t))
            if f_val[i] < f_best:
                f_best = f_val[i]
                best = X[i].copy()
        history.append(f_best)
    return best, f_best, history


if __name__ == "__main__":
    from data_utils import load_binary_mnist
    xtr, ytr, xte, yte = load_binary_mnist(n_train_per_class=200, n_test_per_class=50)
    x_train = xtr[:300]
    x_val = xtr[300:]
    noisy_train = add_gaussian_noise(x_train, 0.5, seed=1)
    noisy_val = add_gaussian_noise(x_val, 0.5, seed=2)

    xt = torch.tensor(x_train).unsqueeze(1)
    nt = torch.tensor(noisy_train).unsqueeze(1)
    xv = torch.tensor(x_val).unsqueeze(1)
    nv = torch.tensor(noisy_val).unsqueeze(1)

    fitness = make_fitness_fn(nt, xt, nv, xv, epochs=8)
    t0 = time.time()
    best, f, hist = run_ehho(fitness, n_pop=4, n_iter=3, seed=0)
    print("E-HHO best", decode(best), "fitness", f, "time", time.time() - t0)
    print("history", hist)
