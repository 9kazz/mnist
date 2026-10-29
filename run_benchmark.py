import argparse
from pathlib import Path
import torch

from src.prepare_data import *
from src.network import *
from src.benchmark import *
import src.visualization as viz

def main(
    device:    torch.device,
    train_log: Path,
    log_dir:   Path | None,
    model_dir: Path,
    num_errors: int | None = None,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")

    _, _, test_dataset = create_mnist_datasets("./data")
    loader = Loader(test_dataset, batch_size=64, shuffle=False)

    model     = load_model(str(model_dir), device=device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if log_dir is None:
        # print results in terminal by default
        run_benchmark(model, loader, criterion, device=device, num_errors=num_errors)
    else:
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        with log_dir.open("w") as benchmark_log:
            run_benchmark(model, loader, criterion, device=device, log_dir=benchmark_log, num_errors=num_errors)

    metrics = viz.read_train_log(train_log)
    viz.show_train_log_plot(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                     type=torch.device, default="cuda",            help="cpu, cuda or cuda:0")
    parser.add_argument("--train-log", "--train_log",   type=Path,         default=None,              help="Path to train log file")
    parser.add_argument("--log-dir",   "--log_dir",     type=Path,         default=None,              help="Output log path; omit to print to stdout")
    parser.add_argument("--model-dir", "--model_dir",   type=Path,         default=Path("model.pt"),  help="Model output path")
    parser.add_argument("--num-errors", "--num_errors", type=int,          default=None,              help="Show the first N incorrect predictions")
    args = parser.parse_args()

    if args.num_errors is not None and args.num_errors < 0:
        parser.error("--num-errors must be non-negative")

    main(
        device=args.device,
        train_log=args.train_log,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        num_errors=args.num_errors,
    )