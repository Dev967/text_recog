import torch
import torch.nn as nn
import torch.nn.functional as F

from CONF import device


class Encoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=1, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)

    def init_hidden(self):
        return torch.zeros((2 if self.bidirectional else 1) * self.num_layers, 1, self.hidden_size, device=device)

    def forward(self, inp, hidden):
        output, hidden = self.gru(inp, hidden)
        return output, hidden


class ConvolutionEncoder(nn.Module):
    def __init__(self):
        super(ConvolutionEncoder, self).__init__()
        self.conv_stack = nn.Sequential(*self.generate_conv(4, 1, 2, (3, 3), 1))
        self.last_conv_idx = 12

    def forward(self, x):
        # print("in -> ", x.shape)
        conv_output = self.conv_stack(x)
        # print("out -> ", conv_output.shape, conv_output.flatten(start_dim=2).shape, self.conv_stack[self.last_conv_idx].weight.data.flatten(start_dim=0).shape)
        # print("kernel: ", self.conv_stack[12].weight.data.flatten().shape)
        return conv_output.flatten(start_dim=2), self.conv_stack[self.last_conv_idx].weight.data.flatten(start_dim=0)

    def generate_conv(self, num_conv, in_channel, out_channel, kernel_size, max_pooling_count):
        conv_arr = []
        for x in range(num_conv):
            conv_arr.append(
                nn.Conv2d(in_channel, out_channel, kernel_size)
            )
            conv_arr.append(
                nn.ReLU(True)
            )
            if x % max_pooling_count == 0: conv_arr.append(nn.MaxPool2d(2, 2))
            conv_arr.append(
                nn.BatchNorm2d(out_channel)
            )
            in_channel = out_channel
            out_channel = out_channel * 2
        return conv_arr

    def init_hidden(self):
        ...


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
    def __init__(self, output_size=80, input_size=80, hidden_size=128, max_length=256, num_layers=1,
                 bidirectional=False):
        super(AttentionDecoder, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_shape = (2 if bidirectional else 1) * num_layers

        self.attn = nn.Linear(input_size + hidden_size * self.hidden_shape, max_length)
        self.attn_combine = nn.Linear(hidden_size * self.num_layers + input_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.out = nn.Linear(hidden_size * self.hidden_shape, output_size)

    def forward(self, sequence, hidden, encoder_outputs):
        hidden = hidden.transpose(1, 0).flatten(start_dim=1).unsqueeze(0)

        concat = torch.cat((sequence[0], hidden[0]), 1)
        attn_weights = F.softmax(self.attn(concat), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.transpose(1, 0))
        output = torch.cat((sequence[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.tanh(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output), dim=2)
        return output, hidden, attn_weights


class ConvolutionalDecoder(nn.Module):
    def __init__(self, output_size=80, input_size=80, hidden_size=128, max_length=256, combine_size=84):
        super(ConvolutionalDecoder, self).__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size

        self.attn = nn.Linear(input_size + hidden_size, max_length)
        self.attn_combine = nn.Linear(combine_size + input_size, input_size)
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
