import argparse
from pathlib import Path
import torch

from src.benchmark import load_model
from src.ideal_digits import generate_ideal_digit
from src.visualization import show_mnist_img


def main(
    model_path: Path,
    device:     torch.device,
    iterations:    int,
    learning_rate: float,
    seed:          int,
    output: Path | None,
) -> None:
    torch.manual_seed(seed)
    model = load_model(str(model_path), device=device)

    images = []
    for digit in range(10):
        image = generate_ideal_digit(model, digit, device, iterations, learning_rate)
        images.append((image, digit))

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)

    show_mnist_img(images, path=output, title="Ideal digits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-file", "--model-dir", type=Path,         required=True,  help="Saved state_dict (.pth or .pt)")
    parser.add_argument("--device",                    type=torch.device, default="cuda", help="cpu, cuda or cuda:0")
    parser.add_argument("--iterations",                type=int,          default=500,    help="Pixel optimization steps")
    parser.add_argument("--lr",                        type=float,        default=0.001,  help="Adam learning rate")
    parser.add_argument("--seed",                      type=int,          default=42,     help="Random seed")
    parser.add_argument("--output",                    type=Path,         default=None,   help="Optional output plot path")
    args = parser.parse_args()

    if args.iterations < 1 or args.lr <= 0:
        parser.error("--iterations and --lr must be positive")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    main(
        model_path=args.model_file,
        device=args.device,
        iterations=args.iterations,
        learning_rate=args.lr,
        seed=args.seed,
        output=args.output,
    )
