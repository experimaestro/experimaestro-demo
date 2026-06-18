"""Monkeypatch torchvision's MNIST with a tiny randomly-generated stand-in.

Loaded as the ``pre_experiment`` hook of ``params-ci.yaml`` so it runs inside
the experiment process *before* the dataset download. The demo code (and the
task subprocesses) stay completely unchanged: we only replace ``MNIST.download``
so it writes a handful of well-formed IDX files instead of fetching the real
~50 MB dataset. The task subprocesses then load those files through the
ordinary (unpatched) ``MNIST(...)`` code path, reading from ``DATAMAESTRO_DIR``.

This keeps the demo pristine — all the test-only machinery lives here.
"""

import os
import struct

import torchvision.datasets as tvd

# Tiny dataset: enough to exercise a training step, small enough to be instant.
_N = 64
_ROWS = _COLS = 28


def _write_idx_images(path: str, n: int) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, n, _ROWS, _COLS))  # magic, n, rows, cols
        f.write(os.urandom(n * _ROWS * _COLS))  # random uint8 pixels


def _write_idx_labels(path: str, n: int) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, n))  # magic, n
        f.write(bytes(i % 10 for i in range(n)))  # labels 0..9


def _fake_download(self) -> None:
    raw = self.raw_folder
    os.makedirs(raw, exist_ok=True)
    _write_idx_images(os.path.join(raw, "train-images-idx3-ubyte"), _N)
    _write_idx_labels(os.path.join(raw, "train-labels-idx1-ubyte"), _N)
    _write_idx_images(os.path.join(raw, "t10k-images-idx3-ubyte"), _N)
    _write_idx_labels(os.path.join(raw, "t10k-labels-idx1-ubyte"), _N)


tvd.MNIST.download = _fake_download
