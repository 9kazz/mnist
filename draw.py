import argparse
from pathlib import Path

from src.draw.draw_digit import *
from src.prepare_data import *

def main(model_path: Path, device: torch.device) -> None:
    model = load_model(str(model_path), device=device)
    root  = create_gui(model, device)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-file", "--model-dir", type=Path,                        help="CNN weights path")
    parser.add_argument("--device",                    type=torch.device, default="cpu", help="cpu, cuda or cuda:0")
    args = parser.parse_args()
    if args.device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu")

    main(model_path=args.model_file, device=args.device)