import torch
import torch.nn as nn

from src.network import Net

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO

from src.prepare_data import Loader
from src.visualization import show_mnist_img

def load_model(path: str, device: torch.device | str) -> Net:
    model = Net().to(device)

    state = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    return model

def run_benchmark(
    model:     nn.Module,
    loader:    Loader,
    criterion: nn.Module,
    device:    torch.device | str,
    log_dir:   TextIO=sys.stdout,
    num_errors: int | None = None,
) -> tuple[float, float]:
    
    if num_errors is not None and num_errors < 0:
        raise ValueError("num_errors must be non-negative or None")

    if len(loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    
    device = torch.device(device)
    model.to(device)
    criterion.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    class_total   = torch.zeros(10, dtype=torch.long, device=device)
    class_correct = torch.zeros(10, dtype=torch.long, device=device)
    errors = []

    with torch.inference_mode():
        for batch_data, batch_ref in loader:
            batch_data = batch_data.to(device)
            batch_ref = batch_ref.to(device)
            pred = model(batch_data)
            pred_label = pred.argmax(dim=1)
            
            loss = criterion(pred, batch_ref)              # mean betch loss 
            total_loss += loss.item() * batch_data.size(0) # total loss of all batch elements

            correct_answers = pred_label == batch_ref
            total_correct += correct_answers.sum().item()
            class_total += torch.bincount(batch_ref, minlength=10)
            class_correct += torch.bincount(batch_ref[correct_answers], minlength=10)

            if num_errors is not None and len(errors) < num_errors:
                indices = (~correct_answers).nonzero(as_tuple=True)[0]
                for index in indices[:num_errors - len(errors)].tolist():
                    image = batch_data[index].detach().cpu().clone()
                    label = batch_ref[index].item()
                    prediction = pred_label[index].item()
                    errors.append((image, f"True: {label}\nPred: {prediction}"))

    mean_loss = total_loss / len(loader.dataset)
    accurancy = total_correct / len(loader.dataset)
    
    print(
        f"Mean loss: {mean_loss:.6f}\nAccurancy: {accurancy:.2%}",
        file=log_dir,
        flush=True
    )

    for digit, (correct, total) in enumerate(zip(class_correct.tolist(), class_total.tolist())):
        if total > 0:
            print(f"Accuracy of {digit}: {correct / total:.2%}", file=log_dir)
        else:
            print(f"Accuracy of {digit}: N/A (no samples)", file=log_dir)
    log_dir.flush()

    if errors:
        show_mnist_img(errors, title="Benchmark errors")

    return mean_loss, accurancy
