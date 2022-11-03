import torch
from CONF import *
import traceback


def train(model, dataloader, loss_fn, optimizer_func, verbose=True):
    model.train()
    loss_arr = []

    for batch, (X, y) in enumerate(dataloader):
        X, y= X.to(device), y.to(device)

        try:
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer_func.zero_grad()
            loss.backward()
            optimizer_func.step()

            loss_arr.append(loss.item())

            if verbose:
                if batch % 100 == 0:
                    print(f'batch: {batch} loss: {loss.item()} [{batch * len(X)} / {len(dataloader.dataset)}]')
        except :
            traceback.print_exc()

    return loss_arr
