import torch
import torch.nn as nn


class NetworkArgs:
    def __init__(self, kernel, depth, channels, stride=1, padding=0, non_linearity=nn.ReLU):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.non_linearity = non_linearity
        self.depth = depth
        self.channels = channels


def gen_conv(conf):
    modules = []
    for x in range(conf.depth):
        in_channel = conf.channels[x]
        out_channel = conf.channels[x+1]
        stack = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, conf.kernel, conf.stride, conf.padding),
            conf.non_linearity()
        )
        modules.append(stack)
    return nn.ModuleList(modules)


class Network(nn.Module):
    def __init__(self, conf):
        super(Network, self).__init__()

        self.stack = gen_conv(conf)
        #convolution output size formula = [(W−K+2P)/S]+1

    def forward(self, x):
        for idx, module in enumerate(self.stack):
            x = module(x)
        return x

