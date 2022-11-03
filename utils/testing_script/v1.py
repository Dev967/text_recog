import torch
import torch.nn as nn


@torch.no_grad()
def test(model, dataloader, loss_fn):
    total_loss = 0
    correct_preds = 0
    total_samples = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        correct_preds += sum(pred.argmax(dim=1) == y)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
    return [total_loss, correct_preds, total_samples]
