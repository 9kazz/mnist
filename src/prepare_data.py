import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import TensorDataset, random_split

RANDOM_SEED = 42

class Loader:
    dataset:    TensorDataset
    batch_size: int
    generator:  torch.Generator
    shuffle:    bool

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 64,
        shuffle:    bool = True,
        rand_seed:  int  = RANDOM_SEED,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not isinstance(dataset, TensorDataset):
            raise TypeError("dataset must be a TensorDataset")
        if len(dataset.tensors) != 2:
            raise ValueError("dataset must contain exactly two tensors")

        data, labels = dataset.tensors

        if data.ndim == 0 or labels.ndim == 0:
            raise ValueError("Data and labels must have a sample dimension")
        if len(data) != len(labels):
            raise ValueError("Data and labels must contain the same number of samples")
        if data.device != labels.device:
            raise ValueError("Data and labels must be on the same device")

        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.generator  = torch.Generator().manual_seed(rand_seed)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        data, labels = self.dataset.tensors

        if self.shuffle:
            indices = torch.randperm(
                len(data),
                generator=self.generator
            ).to(data.device)
        else:
            indices = torch.arange(len(data), device=data.device)

        for start in range(0, len(data), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]

            yield data[batch_indices], labels[batch_indices]

def create_mnist_datasets(
    data_dir: str,
    val_set_size: int = 5000,
    mnist_set_size: int | None = None
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Prepare normalized datasets on CPU; training transfers individual batches."""

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
    )

    # Prepare tensors
    train_images = train_dataset.data.unsqueeze(1).float() / 255.0
    test_images  = test_dataset.data.unsqueeze(1).float()  / 255.0

    # None means use the entire MNIST training dataset
    if mnist_set_size is None:
        mnist_set_size = len(train_dataset)

    if not 0 < val_set_size < mnist_set_size <= len(train_dataset):
        raise ValueError("Invalid dataset sizes")

    # Split into train, validation and unused subsets
    train_divide, val_divide, _ = random_split(
        train_dataset,
        [
            mnist_set_size - val_set_size,
            val_set_size,
            len(train_dataset) - mnist_set_size,
        ],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_tensors = TensorDataset(
        train_images[train_divide.indices],
        train_dataset.targets[train_divide.indices],
    )

    val_tensors = TensorDataset(
        train_images[val_divide.indices],
        train_dataset.targets[val_divide.indices],
    )

    test_tensors = TensorDataset(
        test_images,
        test_dataset.targets,
    )

    return train_tensors, val_tensors, test_tensors
