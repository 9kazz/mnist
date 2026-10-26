import torch
import torch.nn as nn

from src.network import Net

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO

from src.prepare_data import Loader

def load_model(path: str, device: str="cpu") -> Net:
    model = Net()

    state = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model

def run_benchmark(
    model:     nn.Module,
    loader:    Loader,
    criterion: nn.Module,
    log_file:  TextIO=sys.stdout
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_correct = 0

    with torch.inference_mode():
        for batch_data, batch_ref in loader:
            pred = model(batch_data)
            pred_label = pred.argmax(dim=1)
            
            loss = criterion(pred, batch_ref)              # mean betch loss 
            total_loss += loss.item() * batch_data.size(0) # total loss of all batch elements

            correct_answers = pred_label == batch_ref
            total_correct += correct_answers.sum().item()

    assert(len(loader.dataset) != 0)
    mean_loss = total_loss / len(loader.dataset)
    accurancy = total_correct / len(loader.dataset)
    
    print(
        f"Mean loss: {mean_loss:.6f}\nAccurancy: {accurancy:.2%}",
        file=log_file,
        flush=True
    )

    return mean_loss, accurancy
