# QCAE-PSO

Source code for *Metaheuristic-Optimized Quantum Convolutional Autoencoder for
Image Noise Reduction*, which optimizes the hyperparameters of a quantum
convolutional autoencoder (QCAE) with particle swarm optimization (PSO),
compared against grey wolf optimizer (GWO), the bat algorithm (BAT), plain
Harris Hawks Optimization (HHO), and an enhanced HHO variant (E-HHO), on a
reduced binary MNIST (digits 0/1) image-denoising benchmark.

The QAOA-inspired ansatz is reimplemented as a differentiable PyTorch
statevector layer (no Qiskit, no real quantum hardware).

## Repository layout

```
data_utils.py           MNIST loader + Gaussian noise injection
models.py               ClassicalCAE and QCAE (statevector-simulated ansatz)
metaheuristics.py       PSO, GWO, HHO, BAT, E-HHO implementations
run_search.py           Full experiment: search + final retrain/eval -> results/results.json
make_figures.py         Regenerates the convergence/SSIM/training-curve/reconstruction figures
classification_eval.py  Downstream classification-utility experiment (Section 4.3 of the paper)
run_search.log          Console log from the run that produced results/results.json
results/                results.json, classification_results.json (all reported numbers)
figures/                All figures used in the paper
```

Trained model checkpoints (`.pt` files) are not included in this repository;
run `run_search.py` to regenerate them locally.

## Setup

```bash
pip install -r requirements.txt
```

Download the four raw MNIST idx files from http://yann.lecun.com/exdb/mnist/
(or any mirror) and place them under `data/MNIST/`, each in its own subfolder
named after the file, e.g.:

```
data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte
data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte
data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte
data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte
```

Alternatively, point the `MNIST_DIR` environment variable at wherever you
already keep these files.

## Reproducing the results

```bash
python run_search.py          # runs the full search + final retrain/eval, writes results/results.json
python make_figures.py        # regenerates figures/*.png from results/results.json
python classification_eval.py # downstream classification experiment, figures + results/classification_results.json
```

All numbers reported in the paper are produced by these scripts; nothing is
hand-typed.
