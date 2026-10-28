import argparse
from pathlib import Path
import torch

from src.prepare_data import *
from src.network import *
from src.train import *

import torch.optim

def main(
    device:    torch.device,
    log_dir:   Path,
    model_dir: Path,
    val_set_size:   int,
    mnist_set_size: int,
    epoch:          int
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")

    train_dataset, val_dataset, _ = create_mnist_datasets(
        "./data", val_set_size=val_set_size, mnist_set_size=mnist_set_size
    )

    train_loader = Loader(train_dataset, batch_size=64)
    val_loader   = Loader(val_dataset,   batch_size=64, shuffle=False)

    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_dir.open("w", newline="") as train_log:
        train(model, train_loader, val_loader, criterion, optimizer, num_epoch=epoch, device=device, log_dir=train_log)

    save_model(model, str(model_dir))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                             type=torch.device, default="cuda",            help="cpu, cuda or cuda:0")
    parser.add_argument("--log-dir",        "--log_dir",        type=Path,         default=Path("train.csv"), help="Output log path")
    parser.add_argument("--model-dir",      "--model_dir",      type=Path,         default=Path("model.pt"),  help="Model output path")
    parser.add_argument("--val-set-size",   "--val_set_size",   type=int,          default=5000,              help="Validation sample count")
    parser.add_argument("--mnist-set-size", "--mnist_set_size", type=int,          default=60000,             help="Combined training and validation sample count")
    parser.add_argument("--epoch",                              type=int,          default=100,               help="Number of training epoches")
    args = parser.parse_args()

    if not 0 < args.val_set_size < args.mnist_set_size <= 60000:
        parser.error("Expected 0 < val_set_size < mnist_set_size <= 60000")

    main(
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        val_set_size=args.val_set_size,
        mnist_set_size=args.mnist_set_size,
        epoch=args.epoch
    )