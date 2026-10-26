import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1  = nn.Linear(28 * 28, 16)
        self.act1 = nn.ReLU()

        self.layer2  = nn.Linear(16, 16)
        self.act2 = nn.ReLU()

        self.layer3  = nn.Linear(16, 10)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.flatten(start_dim=1) # reshape tesor {batch_sz, 1, 28, 28} into {batch_sz, 28*28}

        output = self.layer1(input)
        output = self.act1(output)

        output = self.layer2(output)
        output = self.act2(output)

        output = self.layer3(output)

        return output