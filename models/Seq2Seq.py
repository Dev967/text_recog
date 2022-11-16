import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AttentionDecoder(nn.Module):
    def __init__(self, output_size=80, input_size=80, hidden_size=128, max_length=256):
        super(AttentionDecoder, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.attn = nn.Linear(input_size + hidden_size, max_length)
        self.attn_combine = nn.Linear(hidden_size + input_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, sequence, hidden, encoder_outputs):
        concat = torch.cat((sequence[0], hidden[0]), 1)
        attn_weights = F.softmax(self.attn(concat), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.transpose(1, 0))
        output = torch.cat((sequence[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)
        return output, hidden, attn_weights
