from abc import ABC, abstractmethod
from pathlib import Path
import logging
from torchvision.datasets import VisionDataset, MNIST
from experimaestro import Meta, Param
from datamaestro import Base, Context
from datamaestro.definitions import AbstractDataset, dataset
from datamaestro.data.ml import Supervised
from datamaestro.download.custom import custom_download


class LabelledImages(Base, ABC):
    @abstractmethod
    def torchvision_dataset(self, **kwargs) -> VisionDataset: ...


class MNISTLabelledImages(LabelledImages):
    root: Meta[Path]
    train: Param[bool]

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return MNIST(self.root, train=self.train, **kwargs)


def download_mnist(context: Context, root: Path, force=False):
    logging.info("Downloading in %s", root)
    for train in [False, True]:
        MNIST(root, train=train, download=True)


@dataset(id="com.lecun.mnist", url="http://yann.lecun.com/exdb/mnist/")
class MNISTDataset(Supervised[LabelledImages, None, LabelledImages]):
    """The MNIST database of handwritten digits."""

    ROOT = custom_download("root", download_mnist)

    @classmethod
    def __create_dataset__(cls, dataset: AbstractDataset):
        return cls.C(
            train=MNISTLabelledImages.C(root=cls.ROOT.prepare(), train=True),
            test=MNISTLabelledImages.C(root=cls.ROOT.prepare(), train=False),
        )
