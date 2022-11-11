import torch
import torch.nn as nn

from CONF import device


class Encoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=128):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def forward(self, inp, hidden):
        output, hidden = self.gru(inp, hidden)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_size=80, input_size=80, hidden_size=128):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, sequence, hidden):
        output, hidden = self.gru(sequence, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden
