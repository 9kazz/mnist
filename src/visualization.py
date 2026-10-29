import torch
import matplotlib.pyplot as plt

from collections.abc import Sequence
import csv

plt.rcParams.update({
    "font.size":       16,      
    "axes.titlesize":  20, 
    "axes.labelsize":  16, 
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 3, 
    "axes.linewidth":  1.5,
})

# Mnist images
#--------------------------------------------------------------------------------

def show_mnist_img(
    objs: Sequence[tuple[torch.Tensor, int | str]],
    path:  str | None = None,
    title: str = "MNIST Dataset",
) -> None:
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
    fig.suptitle(title)

    for idx, ax in enumerate(axes.flat):
        ax.axis("off")
        if idx >= objs_cnt:
            continue

        img, label = objs[idx]

        ax.imshow(img.detach().cpu().reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        if isinstance(label, torch.Tensor):
            label = label.item()
        ax.set_title(label if isinstance(label, str) else f"Label: {label}")

    if path is not None:
        fig.savefig(path, dpi=150)
        
    plt.show()

def read_train_log(path: str) -> dict:
    metrics = {
        "Epoch":         [],
        "TrainLoss":     [],
        "TrainAccuracy": [],
        "ValLoss":       [],
        "ValAccuracy":   [],
    }

    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics["Epoch"].append(int(row["Epoch"]))

            for key in metrics:
                if key == "Epoch":
                    continue
                metrics[key].append(float(row[key]))

    return metrics


# Train loss & accurancy
#--------------------------------------------------------------------------------

def show_train_log_plot(metrics: dict, path: str=None) -> None:
    epochs = metrics["Epoch"]

    fig, (ax_loss, ax_acc) = plt.subplots(
        2, 1,
        figsize=(20, 16),
        layout="constrained",
    )

    # Loss
    ax_loss.plot(epochs, metrics["TrainLoss"], label="Train")
    ax_loss.plot(epochs, metrics["ValLoss"],   label="Validation")

    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, metrics["TrainAccuracy"], label="Train")
    ax_acc.plot(epochs, metrics["ValAccuracy"],   label="Validation")

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    if path != None:
        fig.savefig(path, dpi=150)

    plt.show()