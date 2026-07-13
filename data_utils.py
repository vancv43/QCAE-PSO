"""MNIST digits {0,1} loader + Gaussian noise injection.

Mirrors the reduced-scope protocol in Kit et al. (2025, Mach. Learn.: Sci.
Technol. 6, 045027): binary MNIST subset (digits 0 and 1), Gaussian noise
added only to the training images, clean test images used for evaluation
input (noise then applied at eval time per sigma), pixels normalized to
[0, 1].

Expects the four raw MNIST idx files (train-images-idx3-ubyte,
train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte,
available from http://yann.lecun.com/exdb/mnist/ or any mirror) under
MNIST_DIR, each in its own subfolder named after the file, e.g.:
  <MNIST_DIR>/train-images-idx3-ubyte/train-images-idx3-ubyte

Set the MNIST_DIR environment variable to point at your local copy, or
place the data under ./data/MNIST relative to the repository root.
"""
import numpy as np
import struct
import os

MNIST_DIR = os.environ.get(
    "MNIST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MNIST")
)


def _read_idx(path):
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        n_dims = magic & 0xFF
        # actually standard idx format: first 2 bytes zero, byte3=dtype, byte4=ndims
        f.seek(0)
        zero, dtype_code, ndims = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * ndims, f.read(4 * ndims))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(shape)


def load_binary_mnist(n_train_per_class=250, n_test_per_class=50, seed=0):
    train_images = _read_idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte", "train-images-idx3-ubyte"))
    train_labels = _read_idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte"))
    test_images = _read_idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte"))
    test_labels = _read_idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"))

    rng = np.random.RandomState(seed)

    def _subset(images, labels, n_per_class):
        idx = []
        for c in (0, 1):
            c_idx = np.where(labels == c)[0]
            rng.shuffle(c_idx)
            idx.extend(c_idx[:n_per_class].tolist())
        idx = np.array(idx)
        rng.shuffle(idx)
        x = images[idx].astype(np.float32) / 255.0
        y = labels[idx]
        return x, y

    x_train, y_train = _subset(train_images, train_labels, n_train_per_class)
    x_test, y_test = _subset(test_images, test_labels, n_test_per_class)
    return x_train, y_train, x_test, y_test


def add_gaussian_noise(images, sigma, seed=0):
    rng = np.random.RandomState(seed)
    noisy = images + rng.normal(0.0, sigma, size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


if __name__ == "__main__":
    xtr, ytr, xte, yte = load_binary_mnist()
    print("train", xtr.shape, ytr.shape, "test", xte.shape, yte.shape)
    print("train labels count:", np.bincount(ytr))
    print("pixel range:", xtr.min(), xtr.max())
