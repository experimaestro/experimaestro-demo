from abc import ABC, abstractmethod
from pathlib import Path
from torchvision.datasets import VisionDataset, MNIST
from experimaestro import Meta, Param
from datamaestro import dataset, Base
from datamaestro.data.ml import Supervised
from datamaestro.download.custom import custom_download


class LabelledImages(Base, ABC):
    @abstractmethod
    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        ...


class MNISTLabelledImages(LabelledImages):
    root: Meta[Path]
    train: Param[bool]

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return MNIST(self.root, train=self.train, **kwargs)


def download_mnist(context, root: Path, force=False):
    for train in [False, True]:
        MNIST(root, train=train, download=True)


@dataset(id="com.lecun.mnist")
def mnist() -> Supervised[LabelledImages, None, LabelledImages]:
    """This corresponds to a dataset with an ID `com.lecun.mnist`"""
    root = custom_download(download_mnist)

    return Supervised(
        train=MNISTLabelledImages(root=root, train=True),
        test=MNISTLabelledImages(root=root, train=False),
    )
