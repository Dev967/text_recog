from datasets.MNIST import train_ds, test_ds
from experiments.MNIST_experiment import Experiment
from models.BasicConv import Network
from torch.utils.data import DataLoader
from datasets.MNIST import train_ds
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.data.get_unique_classes import makeUniqueTensor

import torch.nn as nn

import matplotlib.pyplot as plt


if __name__ == '__main__':
    train_dataloader = DataLoader(train_ds, 64, True)
    sample_images, sample_labels = next(iter(train_dataloader))

    exp = Experiment()
    exp.start()
    # exp.test()
    # net = Network()
    # uniq = makeUniqueTensor(sample_images, sample_labels)
    # output = net.conv_stack[0](uniq.unsqueeze(dim=1))
    # output = output.transpose(1, 0)
    # grid1 = make_grid(output[0].unsqueeze(dim=1))
    # grid2 = make_grid(output[1].unsqueeze(dim=1))
    # grid3 = make_grid(output[2].unsqueeze(dim=1))
    # plt.imshow(grid1.permute(1, 2, 0).detach().numpy())
    # plt.show()
    # plt.imshow(grid2.permute(1, 2, 0).detach().numpy())
    # plt.show()
    # plt.imshow(grid3.permute(1, 2, 0).detach().numpy())
    # plt.show()
    # output = net(sample_images)

