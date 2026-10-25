import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# Network architecture
#------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(28 * 28, 16)
        self.act1   = nn.ReLU()

        self.layer2 = nn.Linear(16, 16)
        self.act2   = nn.ReLU()

        self.layer3 = nn.Linear(16, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.flatten(start_dim=1) # reshape tesor {batch_sz, 1, 28, 28} into {batch_sz, 28*28}

        output = self.layer1(input)
        output = self.act1(output)

        output = self.layer2(output)
        output = self.act2(output)

        output = self.layer3(output)

        return output


# Learning
#------------------------------------------------------------------------

def train(
    model: Net,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epoch: int
) -> Net:

    model.train() # switch model on train mode

    for epoch in tqdm(range(num_epoch)):
        epoch_loss = 0.0

        for batch_data, batch_ref in loader:
            optimizer.zero_grad()

            pred = model(batch_data)

            loss = criterion(pred, batch_ref)
            epoch_loss += loss.item() * batch_data.size(0) # total loss of all batch elements

            loss.backward()
            optimizer.step()

        epoch_loss /= len(loader.dataset) # mean loss of all dataset's elements

        print(f"Epoch {epoch + 1}: loss = {epoch_loss:.6f}")

    return model