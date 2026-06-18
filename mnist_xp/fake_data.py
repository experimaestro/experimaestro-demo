"""Tiny randomly-generated stand-in for MNIST.

Used by the CI smoke test (`fake_data: true` in params-ci.yaml) so the demo
experiment can run end-to-end in seconds, on CPU, without downloading the real
dataset. Kept out of ``data.py`` so the teaching code there stays pristine.
"""

from torchvision.datasets import VisionDataset, FakeData
from datamaestro.data.ml import Supervised

from .data import LabelledImages


class FakeLabelledImages(LabelledImages):
    """MNIST-shaped fake data: 1x28x28 images, 10 classes, no download."""

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return FakeData(size=256, image_size=(1, 28, 28), num_classes=10, **kwargs)

    def prepare(self, *args, **kwargs):
        # Nothing to download for fake data.
        return self


def fake_mnist() -> Supervised:
    """A drop-in replacement for ``prepare_dataset(MNISTDataset)``."""
    return Supervised.C(
        train=FakeLabelledImages.C(id="ci.fake.train"),
        test=FakeLabelledImages.C(id="ci.fake.test"),
    )
