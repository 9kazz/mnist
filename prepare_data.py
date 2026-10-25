import torch
import matplotlib.pyplot as plt

from torchvision import datasets
from torch.utils.data import TensorDataset, random_split
from collections.abc import Sequence

# Constants
#------------------------------------------------------------------------
RANDOM_SEED = 42

class Loader:
    dataset: TensorDataset
    data:    list[torch.Tensor]
    labels:  list[torch.Tensor]

    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 64,
        rand_seed:  int = RANDOM_SEED,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if not isinstance(dataset, TensorDataset):
            raise TypeError("dataset must be a TensorDataset")
        if len(dataset.tensors) != 2:
            raise ValueError("dataset must contain exactly two tensors: images and labels")

        self.dataset = dataset
        self.batch_size = batch_size
        self.generator  = torch.Generator().manual_seed(rand_seed)
        self.create_batches()

    def __iter__(self):
            return zip(self.data, self.labels)
    
    def create_batches(self) -> None:
        data, labels = self.dataset.tensors
        if data.ndim == 0 or labels.ndim == 0:
            raise ValueError("Data and labels must have a sample dimension")
        if len(data) != len(labels):
            raise ValueError("Data and labels must contain the same number of samples")
        if data.device != labels.device:
            raise ValueError("Data and labels must be on the same device")

        indices = torch.randperm(len(data), generator=self.generator)
        indices = indices.to(data.device)
        data_batches:  list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []

        for batch_start in range(0, len(data), self.batch_size):
            batch_indices = indices[batch_start : batch_start + self.batch_size]
            data_batches.append(data[batch_indices])
            label_batches.append(labels[batch_indices])

        self.data   = data_batches
        self.labels = label_batches

    def shuffle(self) -> None:
        self.create_batches()

# Prepare data
#------------------------------------------------------------------------
def create_mnist_datasets(
    data_dir: str,
    val_set_size: int = 5000,
    mnist_set_size: int | None = None
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:

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
    test_images  = test_dataset.data.unsqueeze(1).float() / 255.0

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

# Utils
#------------------------------------------------------------------------
def show_mnist(objs: Sequence[tuple[torch.Tensor, int]]) -> None:
    objs_cnt = len(objs)

    if objs_cnt == 0:
        return

    MAX_IMG_IN_LINE = 5
    ncols = min(objs_cnt, MAX_IMG_IN_LINE)
    nrows = (objs_cnt - 1) // ncols + 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        squeeze = False,
        # for nice output
        figsize = (2 * ncols, 2 * nrows),
        layout  = "constrained",
    )
    fig.suptitle("MNIST Dataset")

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= objs_cnt:
            continue

        img, label = objs[idx]

        ax.imshow(img.squeeze(0), cmap="gray")
        ax.set_title(f"Label: {label}")

    plt.show()