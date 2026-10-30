import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1      = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.act1       = nn.ReLU()

        self.conv2      = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.pool2      = nn.MaxPool2d(2)
        self.act2       = nn.ReLU()

        self.conv3      = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.act3       = nn.ReLU()

        self.conv4      = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool4      = nn.MaxPool2d(2)
        self.act4       = nn.ReLU()        

        self.fc5        = nn.Linear(64*7*7, 128)
        self.act5       = nn.ReLU()
        self.dropout    = nn.Dropout(0.3)

        self.fc6        = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.act3(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = self.act4(x)

        x = x.view(-1, 64*7*7)

        x = self.fc5(x)
        x = self.act5(x)
        x = self.dropout(x)

        x = self.fc6(x)

        return x