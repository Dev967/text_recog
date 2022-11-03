import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # self.conv_stack = nn.Sequential(
        #     nn.Conv2d(1, 3, 7),
        #     nn.ReLU(),
        #     nn.Conv2d(3, 6, 7),
        #     nn.ReLU(),
        #     nn.Conv2d(6, 12, 7),
        #     nn.ReLU()
        # )
        self.linear_stack = nn.Sequential(
            nn.Linear(432, 10)
        )
        # self.linear_stack = nn.Sequential(
        #     nn.Linear(1200, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 10),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        output = self.conv_stack(x)
        output = self.linear_stack(output.flatten(start_dim=1))
        return output
