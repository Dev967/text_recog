from CONF import *
import torch.nn as nn


def test_learning_rates(Network, dataloader, learning_rates, train):
    losses = []
    for lr in learning_rates:
        loss_func = nn.CrossEntropyLoss()
        main_net = Network().to(device)
        optimizer_func = torch.optim.SGD(main_net.parameters(), lr=lr)

        loss_arr = train(main_net, dataloader, loss_func, optimizer_func, verbose=False)

        losses.append(loss_arr)
    return losses