import torch.nn as nn
import torch
from utils.transforms.encoding import n_letters
from CONF import *


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, n_layers)
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, n_letters),
            nn.ReLU()
        )

    def init_hidden(self):
        return torch.randn(self.n_layers, batch_size, self.hidden_size)

    def forward(self, x):
        hidden = self.init_hidden()
        output, hidden = self.rnn(x, hidden)
        output = self.linear(output)
        return output
