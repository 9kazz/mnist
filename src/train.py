import torch
import torch.nn as nn

from tqdm.auto import tqdm
from pathlib import Path

import sys
from typing import TextIO

from src.prepare_data import Loader

def train(
    model:  nn.Module,
    train_loader: Loader,
    val_loader:   Loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epoch: int,
    device:    torch.device | str,
    log_dir:   TextIO=sys.stdout
) -> nn.Module:

    # Move the model before constructing the optimizer, not inside train()
    device = torch.device(device)

    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise ValueError("Move the model to device before creating the optimizer")
    criterion.to(device)

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise ValueError("Training and validation datasets must not be empty")
    
    print(
        "Epoch,TrainLoss,TrainAccuracy,ValLoss,ValAccuracy",
        file=log_dir,
        flush=True
    )

    for epoch in tqdm(range(num_epoch)):
        # Train set pass
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_data, batch_ref in train_loader:
            batch_data = batch_data.to(device)
            batch_ref  = batch_ref.to(device)
            
            optimizer.zero_grad()

            pred = model(batch_data)
            pred_label = pred.argmax(dim=1)

            batch_loss  = criterion(pred, batch_ref) # mean batch loss 
            train_loss += batch_loss.item() * batch_data.size(0) # total loss of all batch elements

            correct_answers = pred_label == batch_ref # correct predictions in batch
            train_correct += correct_answers.sum().item()

            batch_loss.backward()
            optimizer.step()

        train_loss /= len(train_loader.dataset) # mean loss of all dataset's elements
        train_accuracy = train_correct / len(train_loader.dataset)

        # Validation set pass
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.inference_mode():
            for batch_data, batch_ref in val_loader:
                batch_data = batch_data.to(device)
                batch_ref  = batch_ref.to(device)

                pred = model(batch_data)
                pred_label = pred.argmax(dim=1)

                batch_loss = criterion(pred, batch_ref)
                val_loss += batch_loss.item() * batch_data.size(0)

                correct_answers = pred_label == batch_ref
                val_correct += correct_answers.sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)
        model.train()

        print(
            f"{epoch + 1},{train_loss:.6f},{train_accuracy:.6f},{val_loss:.6f},{val_accuracy:.6f}",
            file=log_dir, 
            flush=True
        )

    return model

def save_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)    
