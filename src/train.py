import torch
import torch.nn as nn

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO

from src.prepare_data import Loader

def train(
    model:  nn.Module,
    loader: Loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epoch: int,
    log_file:  TextIO=sys.stdout
) -> nn.Module:

    model.train() # switch model on train mode

    for epoch in tqdm(range(num_epoch)):
        epoch_loss = 0.0

        for batch_data, batch_ref in loader:
            optimizer.zero_grad()

            pred = model(batch_data)

            loss = criterion(pred, batch_ref) # mean betch loss 
            epoch_loss += loss.item() * batch_data.size(0) # total loss of all batch elements

            loss.backward()
            optimizer.step()

        assert(len(loader.dataset) != 0)
        epoch_loss /= len(loader.dataset) # mean loss of all dataset's elements

        print(
            f"Epoch {epoch + 1},loss = {epoch_loss:.6f}", 
            file=log_file, 
            flush=True
        )

    return model

def save_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)    
