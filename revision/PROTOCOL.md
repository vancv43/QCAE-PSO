# Revision R1 protocol (SN Computer Science, Reviewer #2 and #3) — locked 2026-10-10 before any run

Original code and results stay untouched at the repository root (results/results.json, results/run_search.log). All new runs are in this folder.
Fixed across all runs: same binary-MNIST split (load seed 0; 300 train / 100 val / 100 test),
same noise seeds, same budget (pop 6, iter 6, 2 trials, 25 search epochs, 80 retrain epochs),
same PSO/GWO/HHO/BAT/E-HHO code and coefficients. No tuning of any optimizer. Every result is
reported, whatever the ranking.

E1a  Retrain-seed robustness (R2 #3). The 7 published final configurations (hyperparameters
     exactly as in results.json) retrained with torch seeds {7, 11, 23, 37, 53}; seed 7 is the
     published seed (reproducibility check). Report mean +- SD per sigma and mean SSIM, and
     per-seed paired differences PSO-HHO, PSO-GWO, PSO-E-HHO.
E1b  Search-seed robustness (R2 #3, and the earlier-ranking-flip concern). Whole pipeline
     (search + retrain) repeated with 3 new search seeds s in {1,2,3}: optimizer seed
     = 1000*s + 100*trial + offset, fitness training seed = s, retrain seed 7. Combined with the
     published run (s=0) this gives 4 independent search repetitions. Report each run's ranking,
     mean +- SD of mean test SSIM per optimizer, and how often each optimizer ranks first.
E2   Widened search space (R2 #2). lr log-uniform [1e-4, 1e-1] (was [1e-4, 1e-2]);
     hidden width {4, 8, 16, 32, 64} (was {4, 8, 16}); weight decay and depth unchanged.
     All five optimizers, published seeds (s=0), same budget. Report selected hp, boundary hits,
     test SSIM per sigma.
E3   QCAE-manual diagnosis (R2 #6). Load the published checkpoints; measure per-pixel std of
     reconstructions across the 100 test inputs, std of the quantum latent z, and fraction of
     angle-encoded inputs with |tanh|>0.99 for every QCAE configuration.
Logging: every fitness evaluation (decoded hp, fitness) is recorded in E1b/E2 so that boundary-hit
rates and search diversity per optimizer can be reported (supports R3 analysis).

## Addendum, added after the locked runs above

E3b  Post-hoc diagnostic added after E3. QCAE-manual retrained with its published hyperparameters for
     80, 200 and 400 epochs to separate under-training from collapse. The first run used one CPU thread
     and is kept in results/superseded/; it was repeated with two threads, like every other revision
     run, and the reported numbers come from that repeat.
E4   Post-hoc test. Exact McNemar test on the paired downstream predictions of QCAE-PSO and QCAE-GWO
     at sigma = 0.5, using the published checkpoints and the classifier set-up of classification_eval.py.
Search-behaviour statistics in aggregate.py are computed over the distinct configurations evaluated in
each trial, because fitness calls include cache hits and repeated leader evaluations.
