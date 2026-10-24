# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""Check every number reported in the paper against the saved result files.

Each check recomputes a value from the JSON files in results/ and revision/results/ and compares it
with the value printed in the paper, at the precision printed in the paper. Nothing is retrained, so the
script runs in about a second. It writes results/verification_report.md and exits with status 1 if any
check fails.

Usage, from the repository root:
    python verify_results.py
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
S = ["0.25", "0.5", "0.75", "1.0"]
OPT = ["GWO", "HHO", "BAT", "E-HHO", "PSO"]


def load(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


R = load("results", "results.json")
C = load("results", "classification_results.json")["results"]
E1A = load("revision", "results", "E1a_retrain_seeds.json")
RUNS = {0: R}
for s in (1, 2, 3):
    RUNS[s] = load("revision", "results", f"E_search_s{s}_orig.json")
WIDE = load("revision", "results", "E_search_s0_wide.json")
E3 = load("revision", "results", "E3_diagnose.json")
E3B = load("revision", "results", "E3b_manual_longer.json")
E4 = load("revision", "results", "E4_mcnemar_pso_gwo.json")

rows = []


def check(section, item, computed, reported):
    """computed is formatted at the paper's precision before comparison."""
    ok = str(computed) == str(reported)
    rows.append((section, item, str(reported), str(computed), "PASS" if ok else "FAIL"))


def f4(x):
    return f"{x:.4f}"


def f3(x):
    return f"{x:.3f}"


def sci(x):
    """Weight decay as printed in Table 1, e.g. 1.5e-4."""
    m, e = f"{x:.1e}".split("e")
    return f"{m}e{int(e)}"


# ---------------------------------------------------------------- Table 1, primary search
T1 = {"GWO": ("0.0100", "1.5e-4", 2, 16, "0.9910", "0.7839", "0.7839", 466),
      "HHO": ("0.0100", "9.2e-5", 3, 8, "0.8800", "0.8745", "0.8745", 379),
      "BAT": ("0.0056", "5.4e-4", 3, 16, "0.9913", "0.9118", "0.9118", 399),
      "E-HHO": ("0.0100", "9.8e-5", 3, 16, "0.7514", "0.8004", "0.7514", 353),
      "PSO": ("0.0100", "1.0e-6", 2, 16, "0.7512", "0.7512", "0.7512", 573)}
for o, (lr, wd, p, w, t1, t2, best, cost) in T1.items():
    d = R["search"][o]
    hp = d["best_hp"]
    check("Table 1", f"{o} learning rate", f4(hp["lr"]), lr)
    check("Table 1", f"{o} weight decay", sci(hp["weight_decay"]), wd)
    check("Table 1", f"{o} depth p", hp["p_layers"], p)
    check("Table 1", f"{o} hidden width", hp["hidden_channels"], w)
    check("Table 1", f"{o} trial 1 fitness", f4(d["trial_fitnesses"][0]), t1)
    check("Table 1", f"{o} trial 2 fitness", f4(d["trial_fitnesses"][1]), t2)
    check("Table 1", f"{o} best fitness", f4(d["best_fitness"]), best)
    check("Table 1", f"{o} search cost, s", round(sum(d["search_cost_seconds"])), cost)

# ---------------------------------------------------------------- Table 2, test SSIM
T2 = {"CCAE-manual": ("0.1545", "0.1607", "0.1662", "0.1703", "0.1629", "0.0719"),
      "QCAE-manual": ("0.0310", "0.0323", "0.0329", "0.0273", "0.0309", "0.0997"),
      "QCAE-GWO": ("0.2590", "0.2004", "0.1753", "0.1528", "0.1969", "0.0327"),
      "QCAE-HHO": ("0.3002", "0.2539", "0.2530", "0.2321", "0.2598", "0.0386"),
      "QCAE-BAT": ("0.1090", "0.1458", "0.1176", "0.1219", "0.1236", "0.0443"),
      "QCAE-E-HHO": ("0.2258", "0.2152", "0.2410", "0.2332", "0.2288", "0.0311"),
      "QCAE-PSO": ("0.3167", "0.2684", "0.2627", "0.2710", "0.2797", "0.0292")}
for m, vals in T2.items():
    d = R["final"][m]
    for s, v in zip(S, vals[:4]):
        check("Table 2", f"{m} SSIM at sigma={s}", f4(d["ssim_by_sigma"][s]), v)
    check("Table 2", f"{m} mean SSIM", f4(d["mean_ssim"]), vals[4])
    check("Table 2", f"{m} final training loss", f4(d["final_train_loss"]), vals[5])

# ---------------------------------------------------------------- Table 4, five retraining seeds
T4 = {"CCAE-manual": ("0.1589", "0.0099", "0.1487", "0.1733"), "QCAE-manual": ("0.0246", "0.0044", "0.0198", "0.0300"),
      "QCAE-GWO": ("0.1943", "0.0142", "0.1732", "0.2129"), "QCAE-HHO": ("0.2243", "0.0488", "0.1762", "0.2980"),
      "QCAE-BAT": ("0.1425", "0.0081", "0.1344", "0.1518"), "QCAE-E-HHO": ("0.2500", "0.0724", "0.1873", "0.3748"),
      "QCAE-PSO": ("0.2907", "0.0257", "0.2695", "0.3341")}
seed_mean = {m: np.array([r["mean_ssim"] for r in E1A[m].values()]) for m in E1A}
for m, (mu, sd, lo, hi) in T4.items():
    x = seed_mean[m]
    check("Table 4", f"{m} mean over seeds", f4(x.mean()), mu)
    check("Table 4", f"{m} SD over seeds", f4(x.std(ddof=1)), sd)
    check("Table 4", f"{m} lowest seed", f4(x.min()), lo)
    check("Table 4", f"{m} highest seed", f4(x.max()), hi)
for o, n in (("GWO", 5), ("HHO", 5), ("BAT", 5), ("E-HHO", 4)):
    check("Text, retraining seeds", f"seeds in which PSO scored above {o}",
          int((seed_mean["QCAE-PSO"] > seed_mean[f"QCAE-{o}"]).sum()), n)
check("Text, retraining seeds", "mean margin of PSO over HHO", f3((seed_mean["QCAE-PSO"] - seed_mean["QCAE-HHO"]).mean()), "0.066")
check("Text, retraining seeds", "mean margin of PSO over GWO", f3((seed_mean["QCAE-PSO"] - seed_mean["QCAE-GWO"]).mean()), "0.096")
check("Text, retraining seeds", "best E-HHO seed", f3(seed_mean["QCAE-E-HHO"].max()), "0.375")
rep = {m: abs(E1A[m]["7"]["mean_ssim"] - R["final"][m]["mean_ssim"]) for m in E1A}
check("Text, retraining seeds", "seed 7 reproduces CCAE-manual, QCAE-manual, PSO and GWO within 0.002",
      all(rep[m] <= 0.002 for m in ("CCAE-manual", "QCAE-manual", "QCAE-PSO", "QCAE-GWO")), True)
others = [rep[m] for m in ("QCAE-E-HHO", "QCAE-BAT", "QCAE-HHO")]
check("Text, retraining seeds", "seed 7 differs for E-HHO, BAT and HHO by", f"{min(others):.3f} to {max(others):.3f}", "0.011 to 0.023")

# ---------------------------------------------------------------- Table 5, four search repetitions
T5 = {"GWO": ("0.1969", "0.2189", "0.3092", "0.2630", "0.2470", "0.0498", 1),
      "HHO": ("0.2598", "0.1547", "0.2208", "0.0785", "0.1784", "0.0795", 0),
      "BAT": ("0.1236", "0.1895", "0.1880", "0.1762", "0.1693", "0.0311", 0),
      "E-HHO": ("0.2288", "0.2996", "0.2460", "0.2377", "0.2530", "0.0318", 1),
      "PSO": ("0.2797", "0.1694", "0.2741", "0.2886", "0.2529", "0.0560", 2)}
rep_ssim = {o: np.array([RUNS[s]["final"][f"QCAE-{o}"]["mean_ssim"] for s in range(4)]) for o in OPT}
winners = [max(OPT, key=lambda o: rep_ssim[o][s]) for s in range(4)]
for o, vals in T5.items():
    for s in range(4):
        check("Table 5", f"{o} run {s}", f4(rep_ssim[o][s]), vals[s])
    check("Table 5", f"{o} mean", f4(rep_ssim[o].mean()), vals[4])
    check("Table 5", f"{o} SD", f4(rep_ssim[o].std(ddof=1)), vals[5])
    check("Table 5", f"{o} runs won", winners.count(o), vals[6])
top = [rep_ssim[o].mean() for o in ("PSO", "GWO", "E-HHO")]
check("Text, search repetitions", "PSO, GWO and E-HHO means lie within", f3(max(top) - min(top)), "0.006")
others = np.mean([rep_ssim["PSO"][s] for s in (0, 2, 3)])
check("Text, search repetitions", "PSO run 1 below its other runs, percent", round(100 * (1 - rep_ssim["PSO"][1] / others)), 40)
check("Text, search repetitions", "PSO hidden width in both run 1 trials",
      sorted({e["hp"]["hidden_channels"] for t in RUNS[1]["search"]["PSO"]["evaluations"][:2] for e in t[-6:]}), [4])

# fitness evaluations: GWO calls fitness() 54 times per trial, but 12 calls re-read the cached beta and delta
# leaders, so no optimizer except E-HHO trains more than 42 distinct configurations per trial
for o in OPT:
    distinct = [len({tuple(np.round(e["vec"], 6)) for e in t}) for s in (1, 2, 3) for t in RUNS[s]["search"][o]["evaluations"]]
    calls = {len(t) for s in (1, 2, 3) for t in RUNS[s]["search"][o]["evaluations"]}
    limit = 48 if o == "E-HHO" else 42
    check("Text, budget", f"{o}: distinct configurations trained per trial at most {limit}", max(distinct) <= limit, True)
    if o == "E-HHO":
        check("Text, budget", "E-HHO fitness calls per trial", calls, {48})

# share of distinct configurations at the learning-rate ceiling, runs 1 to 3
SHARE = {"PSO": 59, "GWO": 35, "HHO": 36, "E-HHO": 23, "BAT": 9}
for o, pct in SHARE.items():
    uniq = []
    for s in (1, 2, 3):
        for t in RUNS[s]["search"][o]["evaluations"]:
            seen = {}
            for e in t:
                seen.setdefault(tuple(np.round(e["vec"], 6)), e)
            uniq.extend(seen.values())
    check("Text, search behaviour", f"{o} share of distinct configurations at the maximum learning rate, percent",
          round(100 * np.mean([e["vec"][0] >= 0.999 for e in uniq])), pct)

cost = {o: sum(R["search"][o]["search_cost_seconds"]) for o in OPT}
check("Text, search cost", "PSO search cost above the mean of the other four, percent",
      round(100 * (cost["PSO"] / np.mean([cost[o] for o in OPT if o != "PSO"]) - 1)), 44)
check("Abstract", "PSO mean test SSIM above CCAE-manual, percent",
      round(100 * (R["final"]["QCAE-PSO"]["mean_ssim"] / R["final"]["CCAE-manual"]["mean_ssim"] - 1)), 72)

# ---------------------------------------------------------------- Table 6, widened search space
T6 = {"GWO": ("0.0582", 32, 2, "0.4519", "0.5047", "0.4093", "0.3749", "0.4352"),
      "HHO": ("0.0226", 32, 3, "0.5917", "0.5608", "0.3718", "0.2274", "0.4379"),
      "BAT": ("0.0372", 16, 3, "0.5798", "0.5175", "0.4272", "0.2616", "0.4465"),
      "E-HHO": ("0.0788", 8, 3, "0.5184", "0.5046", "0.4422", "0.3516", "0.4542"),
      "PSO": ("0.1000", 16, 1, "0.5751", "0.6361", "0.5946", "0.5153", "0.5803")}
for o, v in T6.items():
    hp = WIDE["search"][o]["best_hp"]
    d = WIDE["final"][f"QCAE-{o}"]
    check("Table 6", f"{o} learning rate", f4(hp["lr"]), v[0])
    check("Table 6", f"{o} hidden width", hp["hidden_channels"], v[1])
    check("Table 6", f"{o} depth p", hp["p_layers"], v[2])
    for s, x in zip(S, v[3:7]):
        check("Table 6", f"{o} SSIM at sigma={s}", f4(d["ssim_by_sigma"][s]), x)
    check("Table 6", f"{o} mean SSIM", f4(d["mean_ssim"]), v[7])
ratio = [WIDE["final"][f"QCAE-{o}"]["mean_ssim"] / R["final"][f"QCAE-{o}"]["mean_ssim"] for o in OPT]
check("Abstract and text, widened space", "SSIM gain factor over the original space", f"{min(ratio):.1f} to {max(ratio):.1f}", "1.7 to 3.6")
lrs = [WIDE["search"][o]["best_hp"]["lr"] for o in OPT]
check("Text, widened space", "selected learning rates", f"{min(lrs):.3f} to {max(lrs):.1f}", "0.023 to 0.1")

# ---------------------------------------------------------------- Table 7, downstream classification
T7 = {"Noisy (no denoising)": ("0.950", "1.000", "0.900", "0.947", "1.000", "1.000"),
      "CCAE-manual": ("0.460", "0.479", "0.900", "0.625", "0.099", "0.327"),
      "QCAE-manual": ("0.500", "--", "0.000", "--", "0.553", "0.584"),
      "QCAE-GWO": ("0.610", "0.622", "0.560", "0.589", "0.637", "0.572"),
      "QCAE-HHO": ("0.510", "0.511", "0.480", "0.495", "0.541", "0.568"),
      "QCAE-BAT": ("0.560", "0.594", "0.380", "0.463", "0.589", "0.590"),
      "QCAE-E-HHO": ("0.540", "0.556", "0.400", "0.465", "0.562", "0.585"),
      "QCAE-PSO": ("0.520", "0.521", "0.500", "0.510", "0.564", "0.530")}
for m, v in T7.items():
    d = C[m]
    undefined = d["confusion_matrix"][0][1] + d["confusion_matrix"][1][1] == 0   # no positive predictions
    for k, x in zip(("accuracy", "precision", "recall", "f1", "auc", "average_precision"), v):
        val = "--" if undefined and k in ("precision", "f1") else f3(d[k])
        check("Table 7", f"{m} {k}", val, x)
searched = [f"QCAE-{o}" for o in OPT]
for k in ("accuracy", "precision", "recall", "f1", "auc"):
    check("Abstract and text, downstream", f"best searched configuration by {k}", max(searched, key=lambda m: C[m][k]), "QCAE-GWO")
check("Text, downstream", "best average precision", max(searched + ["QCAE-manual"], key=lambda m: C[m]["average_precision"]), "QCAE-BAT")
check("Text, downstream", "PSO leads on no downstream metric",
      not any(max(searched, key=lambda m: C[m][k]) == "QCAE-PSO" for k in ("accuracy", "precision", "recall", "f1", "auc", "average_precision")), True)
check("Text, downstream", "McNemar p-value, PSO against GWO", f"{E4['exact_mcnemar_p']:.2f}", "0.21")
check("Text, downstream", "images correct only after PSO / only after GWO",
      f"{E4['pso_right_gwo_wrong']} / {E4['gwo_right_pso_wrong']}", "16 / 25")

# ---------------------------------------------------------------- QCAE-manual diagnosis
check("Text, diagnosis", "QCAE-manual latent angles beyond |tanh| = 0.99, percent below 1",
      E3["QCAE-manual"]["frac_saturated_tanh_gt_0.99"] < 0.01, True)
check("Text, diagnosis", "QCAE-manual mean SSIM at 80, 200 and 400 epochs",
      ", ".join(f3(E3B[k]["mean_ssim"]) for k in ("80", "200", "400")), "0.030, 0.085, 0.128")

# ---------------------------------------------------------------- report
n_fail = sum(r[4] == "FAIL" for r in rows)
sha = {}
for rel in ("results/results.json", "results/classification_results.json", "revision/results/E1a_retrain_seeds.json",
            "revision/results/E_search_s1_orig.json", "revision/results/E_search_s2_orig.json",
            "revision/results/E_search_s3_orig.json", "revision/results/E_search_s0_wide.json",
            "revision/results/E3_diagnose.json", "revision/results/E3b_manual_longer.json",
            "revision/results/E4_mcnemar_pso_gwo.json"):
    with open(os.path.join(ROOT, rel), "rb") as f:
        sha[rel] = hashlib.sha256(f.read()).hexdigest()
out = ["# Verification of the reported numbers", "",
       f"{len(rows)} checks, {len(rows) - n_fail} passed, {n_fail} failed. Generated by `verify_results.py` from the files below.", "",
       "| Source file | SHA-256 |", "|---|---|"] + [f"| `{k}` | `{v}` |" for k, v in sha.items()] + [
       "", "| Where in the paper | Quantity | Paper | Recomputed | Result |", "|---|---|---|---|---|"] + [
       f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
with open(os.path.join(ROOT, "results", "verification_report.md"), "w") as f:
    f.write("\n".join(out) + "\n")
for r in rows:
    if r[4] == "FAIL":
        print("FAIL", r)
print(f"{len(rows)} checks, {len(rows) - n_fail} passed, {n_fail} failed -> results/verification_report.md")
sys.exit(1 if n_fail else 0)
