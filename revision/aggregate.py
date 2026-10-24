# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""Aggregate the revision results into results/summary.md, the source of the robustness numbers in the paper."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(HERE)

S = ["0.25", "0.5", "0.75", "1.0"]
OPT = ["GWO", "HHO", "BAT", "E-HHO", "PSO"]
lines = []
P = lines.append

# E1a
e1a = json.load(open("results/E1a_retrain_seeds.json"))
P("## E1a retrain seeds (published hp), mean +- SD over seeds")
P("| config | " + " | ".join(f"s={s}" for s in S) + " | mean SSIM | seeds |")
for tag, runs in e1a.items():
    vals = np.array([[r["ssim_by_sigma"][s] for s in S] for r in runs.values()])
    m = vals.mean(1)
    cells = [f"{vals[:, j].mean():.4f} ± {vals[:, j].std(ddof=1):.4f}" for j in range(4)]
    P(f"| {tag} | " + " | ".join(cells) + f" | {m.mean():.4f} ± {m.std(ddof=1):.4f} | "
      + ", ".join(f"{x:.4f}" for x in m) + " |")
pso = np.array([e1a["QCAE-PSO"][k]["mean_ssim"] for k in e1a["QCAE-PSO"]])
for o in ["HHO", "GWO", "E-HHO", "BAT"]:
    other = np.array([e1a[f"QCAE-{o}"][k]["mean_ssim"] for k in e1a["QCAE-PSO"]])
    d = pso - other
    P(f"PSO - {o}: per-seed diffs {np.round(d, 4).tolist()}, mean {d.mean():.4f}, PSO ahead in {(d > 0).sum()}/{len(d)}")

# E1b (+ published s=0)
orig = json.load(open(os.path.join(ROOT, "results", "results.json")))
runs = {0: orig}
for s in [1, 2, 3]:
    f = f"results/E_search_s{s}_orig.json"
    if os.path.exists(f):
        runs[s] = json.load(open(f))
P("\n## E1b search repetitions (s=0 is the published run)")
P("| s | " + " | ".join(OPT) + " | winner |")
tab = {o: [] for o in OPT}
wins = {o: 0 for o in OPT}
for s, d in runs.items():
    row = {o: d["final"][f"QCAE-{o}"]["mean_ssim"] for o in OPT}
    w = max(row, key=row.get)
    wins[w] += 1
    for o in OPT:
        tab[o].append(row[o])
    P(f"| {s} | " + " | ".join(f"{row[o]:.4f}" for o in OPT) + f" | {w} |")
P("| mean ± SD | " + " | ".join(f"{np.mean(tab[o]):.4f} ± {np.std(tab[o], ddof=1):.4f}" for o in OPT) + " | |")
P("wins: " + str(wins))
ccae = orig["final"]["CCAE-manual"]["mean_ssim"]
P("beats CCAE-manual (0.1629) count: " + str({o: int(sum(v > ccae for v in tab[o])) for o in OPT}))
P("\nselected hp per run:")
for s, d in runs.items():
    P(f"s={s}: " + "; ".join(f"{o} lr={d['search'][o]['best_hp']['lr']:.4g} wd={d['search'][o]['best_hp']['weight_decay']:.2g} "
                          f"p={d['search'][o]['best_hp']['p_layers']} w={d['search'][o]['best_hp']['hidden_channels']} "
                          f"fit={d['search'][o]['best_fitness']:.4f}" for o in OPT))

# boundary statistics over the distinct configurations evaluated in each trial (E1b, s=1..3)
P("\n## Search behaviour, distinct configurations per trial (s=1..3, original space)")
P("| opt | fitness calls/trial | distinct configs/trial | share at max lr | share with width 16 | share at min wd |")
for o in OPT:
    calls, uniq = [], []
    for s in [1, 2, 3]:
        if s not in runs:
            continue
        for trial in runs[s]["search"][o]["evaluations"]:
            calls.append(len(trial))
            seen = {}
            for e in trial:
                seen.setdefault(tuple(np.round(e["vec"], 6)), e)
            uniq.extend(seen.values())
    n_trials = len(calls)
    if n_trials:
        P(f"| {o} | {np.mean(calls):.0f} | {len(uniq) / n_trials:.1f} | "
          f"{np.mean([e['vec'][0] >= 0.999 for e in uniq]):.2f} | "
          f"{np.mean([e['hp']['hidden_channels'] == 16 for e in uniq]):.2f} | "
          f"{np.mean([e['vec'][1] <= 0.001 for e in uniq]):.2f} |")

# E2 wide
f = "results/E_search_s0_wide.json"
if os.path.exists(f):
    d = json.load(open(f))
    P("\n## E2 widened space (lr up to 1e-1, width up to 64), seed s=0")
    P("| opt | lr | wd | p | width | best fitness | " + " | ".join(f"s={s}" for s in S) + " | mean | orig-space mean |")
    for o in OPT:
        hp = d["search"][o]["best_hp"]
        r = d["final"][f"QCAE-{o}"]
        P(f"| {o} | {hp['lr']:.4g} | {hp['weight_decay']:.2g} | {hp['p_layers']} | {hp['hidden_channels']} | "
          f"{d['search'][o]['best_fitness']:.4f} | " + " | ".join(f"{r['ssim_by_sigma'][s]:.4f}" for s in S)
          + f" | {r['mean_ssim']:.4f} | {orig['final'][f'QCAE-{o}']['mean_ssim']:.4f} |")
    costs = {o: sum(d["search"][o]["search_cost_seconds"]) for o in OPT}
    P("E2 search cost s: " + str({o: round(c) for o, c in costs.items()}))

open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
