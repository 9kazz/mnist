import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from collections.abc import Sequence

# Constants
#------------------------------------------------------------------------
RANDOM_SEED = 42

# Prepare data
#------------------------------------------------------------------------
def create_mnist_datasets(
    data_dir,
    val_set_size: int = 5000,
    mnist_set_size: int | None = None
) -> tuple[Dataset, Dataset, Dataset]:

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # None means use the entire MNIST training dataset
    if mnist_set_size is None:
        mnist_set_size = len(train_dataset)

    if not 0 < val_set_size < mnist_set_size <= len(train_dataset):
        raise ValueError("Invalid dataset sizes")

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Select a random subset of the original training dataset
    selected_dataset, _ = random_split(
        train_dataset,
        [mnist_set_size, len(train_dataset) - mnist_set_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    # Split the selected subset into train and validation
    train_dataset, val_dataset = random_split(
        selected_dataset,
        [mnist_set_size - val_set_size, val_set_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset,
                       train_batch_size: int = 64, eval_batch_size: int = 64
                      ) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size = train_batch_size,
        shuffle    = True,
        generator = torch.Generator().manual_seed(RANDOM_SEED),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = eval_batch_size,
        shuffle    = False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = eval_batch_size,
        shuffle    = False,
    )
    return train_loader, val_loader, test_loader

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