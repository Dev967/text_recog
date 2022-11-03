import torch
from CONF import *


def build_confusion_matrix(main_net, dataloader, classes):
    predictions = torch.Tensor([]).long().to(device)
    labels = torch.Tensor([]).long().to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.long().to(device)
        pred = main_net(X).long()

        labels = torch.cat((labels, y), 0)
        predictions = torch.cat((predictions, pred.argmax(dim=1)), 0)

    size = len(classes)
    temp = [0 for x in range(size)]
    matrix = [temp for x in range(size)]
    matrix = torch.Tensor(matrix).long().to(device)

    for idx, label in enumerate(labels):
        matrix[label][predictions[idx]] += 1

    return matrix.detach().cpu()
