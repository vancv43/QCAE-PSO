# QCAE-PSO

Code, logs, trained checkpoints and result files for the paper

> **Metaheuristic-Optimized Quantum Convolutional Autoencoder for Image Noise Reduction**
> Van Van Chau, Hao Anh Nguyen, Tuan Minh Nguyen, Chinda Thorn, Phayung Meesad (under review).

The paper tunes the hyperparameters of a quantum convolutional autoencoder (QCAE) with particle swarm
optimization (PSO) and compares PSO with the grey wolf optimizer (GWO), the bat algorithm (BAT), plain
Harris hawks optimization (HHO) and an enhanced HHO variant (E-HHO). The benchmark is image denoising on
a reduced binary MNIST set (digits 0 and 1) under Gaussian noise. Every number in the paper is produced
by the scripts in this repository, and the raw result files are included.

## Method in brief

- **Model.** A convolutional encoder compresses a 28×28 image to 4 values. Batch normalization and
  `tanh` map them to angles in [−π, π], which are angle-encoded on 4 qubits. A QAOA-inspired ansatz with
  `p` layers applies a ring of CNOT gates and trainable R<sub>Y</sub>/R<sub>Z</sub> rotations. The
  per-qubit Pauli-Z expectations go to a mirrored convolutional decoder. The circuit is simulated as an
  exact statevector in PyTorch, so gradients come from automatic differentiation rather than the
  parameter-shift rule. No quantum hardware or Qiskit is used.
- **Baseline.** A classical CAE with the same encoder and decoder and a 4-dimensional latent space.
- **Search space.** Each optimizer searches a 4-D vector in [0, 1]<sup>4</sup>. The vector is decoded
  to a learning rate, log-uniform in [10<sup>−4</sup>, 10<sup>−2</sup>]; a weight decay, log-uniform in
  [10<sup>−6</sup>, 10<sup>−2</sup>]; an ansatz depth p ∈ {1, 2, 3}; and a hidden width ∈ {4, 8, 16}.
- **Fitness.** 1 − mean SSIM on a 100-image validation split, after 25 training epochs at σ = 0.5.
- **Protocol.** Population 6, 6 iterations, 2 independent trials per optimizer. The best configuration
  is retrained for 80 epochs on train + validation (400 images) and evaluated once on 100 held-out test
  images at σ ∈ {0.25, 0.5, 0.75, 1.0}.

## Main results

Test-set SSIM of the primary run, from `results/results.json` (Table 2 of the paper):

| Model | σ=0.25 | σ=0.5 | σ=0.75 | σ=1.0 | Mean SSIM |
|---|---|---|---|---|---|
| CCAE-manual | 0.1545 | 0.1607 | 0.1662 | 0.1703 | 0.1629 |
| QCAE-manual | 0.0310 | 0.0323 | 0.0329 | 0.0273 | 0.0309 |
| QCAE-GWO | 0.2590 | 0.2004 | 0.1753 | 0.1528 | 0.1969 |
| QCAE-HHO | 0.3002 | 0.2539 | 0.2530 | 0.2321 | 0.2598 |
| QCAE-BAT | 0.1090 | 0.1458 | 0.1176 | 0.1219 | 0.1236 |
| QCAE-E-HHO | 0.2258 | 0.2152 | 0.2410 | 0.2332 | 0.2288 |
| QCAE-PSO | 0.3167 | 0.2684 | 0.2627 | 0.2710 | 0.2797 |

The revision experiments in `revision/` test how far this ranking holds:

| Experiment | Outcome |
|---|---|
| Five retraining seeds, published hyperparameters fixed | The PSO configuration has the highest mean SSIM, 0.291 ± 0.026. It is ahead of the GWO, HHO and BAT configurations in all five seeds and ahead of E-HHO in four. |
| Four independent repetitions of the whole search | Mean SSIM: PSO 0.253 ± 0.056, E-HHO 0.253 ± 0.032, GWO 0.247 ± 0.050, HHO 0.178 ± 0.080, BAT 0.169 ± 0.031. PSO won 2 of the 4 runs; PSO, E-HHO and GWO cannot be separated. |
| Search with the learning rate widened to [10<sup>−4</sup>, 10<sup>−1</sup>] and width to {4, …, 64}, one run | Test SSIM of every optimizer rose by a factor of 1.7 to 3.6; PSO reached 0.580, the others 0.435 to 0.454. The original learning-rate bound was too narrow. |
| Downstream classification, a clean-trained logistic regression on the σ = 0.5 reconstructions | Every reconstruction is classified worse than the raw noisy image. GWO has the highest accuracy, 0.61 against 0.52 for PSO; an exact McNemar test gives p = 0.21. |

## Repository layout

```
models.py               ClassicalCAE and QCAE with the statevector-simulated ansatz
metaheuristics.py       PSO, GWO, HHO, BAT and E-HHO, the search space decoder and the fitness function
data_utils.py           Binary MNIST loader and Gaussian noise injection
run_search.py           Search, final retraining and test evaluation -> results/results.json
classification_eval.py  Downstream classification metrics -> results/classification_results.json
results/
  results.json                  search histories, selected hyperparameters, test SSIM, training curves
  classification_results.json   confusion matrices, ROC and precision-recall data
  run_search.log                console log of the run that produced results.json
  checkpoints/                  the seven trained final models used in the paper
figures/
  Fig1-Fig7, Algorithm1 PDF     all figures of the paper
revision/
  PROTOCOL.md                   experiment protocol, fixed before the revision runs, plus an addendum
  run_revision.py               E1a retraining seeds, E1b search repetitions, E2 widened space, E3 diagnosis
  e3b_longer.py                 E3b longer training of QCAE-manual
  mcnemar_pso_gwo.py            E4 McNemar test, PSO against GWO
  aggregate.py                  builds results/summary.md from all revision result files
  logs/, results/               console logs and JSON outputs of every revision run
```

## Setup

Python 3.10 and the packages in `requirements.txt`, which lists the exact versions used:

```bash
pip install -r requirements.txt
```

Download the four MNIST idx files and place each one in a folder named after the file:

```
data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte
data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte
data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

To keep the data somewhere else, set the `MNIST_DIR` environment variable to that folder.

## Reproducing the results

Main experiment, about 40 minutes on an 8-core CPU:

```bash
python run_search.py           # results/results.json and results/checkpoints/
python classification_eval.py  # results/classification_results.json
```

`classification_eval.py` reads the checkpoints in `results/checkpoints/`, so it reproduces the paper's
Table 7 without rerunning the search.

Revision experiments, run from the repository root:

```bash
python revision/run_revision.py retrain_seeds --threads 2          # E1a, about 10 minutes
python revision/run_revision.py search --s 1 --space orig --threads 2  # E1b, also s = 2 and 3, about 25 minutes each
python revision/run_revision.py search --s 0 --space wide --threads 2  # E2, about 35 minutes
python revision/run_revision.py diagnose                           # E3, seconds
python revision/e3b_longer.py                                      # E3b, a few minutes
python revision/mcnemar_pso_gwo.py                                 # E4, seconds
python revision/aggregate.py                                       # revision/results/summary.md
```

## Reproducibility notes

- All data splits, noise realizations, optimizer streams and training seeds are fixed in the code.
- QCAE training at a learning rate of 0.01 is sensitive to the order of floating-point reductions,
  which depends on the number of CPU threads. Retraining the published configurations with the same
  seed reproduced CCAE-manual, QCAE-manual, PSO and GWO to within 0.002 SSIM, and E-HHO, BAT and HHO
  to within 0.011 to 0.023. The multi-seed means in `revision/` are therefore the more reliable
  estimate. All revision runs used two threads.
- The checkpoints in `results/checkpoints/` reproduce Table 2 and Table 7 exactly.

## Implementation notes

- Plain HHO and E-HHO keep the exploration phase of Heidari et al. (2019). They replace its four
  besiege strategies, including the Lévy-flight dives, with the single move
  `x = x_rabbit − E·|J·x_rabbit − x|`.
- E-HHO adds chaotic logistic-map initialization, an oscillating escaping-energy schedule and elite
  opposition-based learning. The opposition step costs one fitness evaluation per iteration, so E-HHO
  uses 48 evaluations per trial against 42 for the other optimizers.
- Fitness values are cached per configuration, and every candidate is trained with the same seed, so
  the fitness of a configuration is deterministic within a search.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{chau2026qcaepso,
  title   = {Metaheuristic-Optimized Quantum Convolutional Autoencoder for Image Noise Reduction},
  author  = {Van Chau, Van and Nguyen, Hao Anh and Nguyen, Tuan Minh and Thorn, Chinda and Meesad, Phayung},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Corresponding author: Tuan Minh Nguyen, Posts and Telecommunications Institute of Technology, Viet Nam.
