import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1      = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=1) # may use padding="same"
        self.conv_act1  = nn.ReLU()
        self.pool1      = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv2      = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=4, stride=1) # may use padding="same"
        self.conv_act2  = nn.ReLU()
        self.pool2      = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1     = nn.Linear(8*11*11, 16)
        self.fc_act1 = nn.ReLU()

        self.fc2     = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv_act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv_act2(x)
        x = self.pool2(x)

        x = x.view(-1, 8*11*11)
        x = self.fc1(x)
        x = self.fc_act1(x)

        x = self.fc2(x)
        
        return x