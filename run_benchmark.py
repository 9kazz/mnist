import argparse
from pathlib import Path
import torch

from src.prepare_data import *
from src.network import *
from src.benchmark import *

def main(
    device: torch.device,
    log_dir:   Path | None,
    model_dir: Path,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu")

    _, _, test_dataset = create_mnist_datasets("./data")
    loader = Loader(test_dataset, batch_size=64, shuffle=False)

    model     = load_model(str(model_dir), device=device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if log_dir is None:
        # print results in terminal by default
        run_benchmark(model, loader, criterion, device=device)
    else:
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        with log_dir.open("w") as benchmark_log:
            run_benchmark(model, loader, criterion, device=device, log_dir=benchmark_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device",                   type=torch.device, default="cuda",            help="cpu, cuda or cuda:0")
    parser.add_argument("--log-dir",   "--log_dir",   type=Path,         default=None,              help="Output log path; omit to print to stdout")
    parser.add_argument("--model-dir", "--model_dir", type=Path,         default=Path("model.pt"),  help="Model output path")
    args = parser.parse_args()

    main(
        device=args.device,
        log_dir=args.log_dir,
        model_dir=args.model_dir
    )