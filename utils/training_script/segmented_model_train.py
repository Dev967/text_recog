import random
import traceback

import torch

from datasets.IAM_words import lang


def train(encoder, decoder, dataloader, loss_fn, optimizer_func, verbose=True):
    batch_loss = []
    total = len(dataloader.dataset)
    print(f'Training Dataset Size: {total}')
    idx = 0
    for batch, (X, y) in enumerate(dataloader):
        idx += 1
        # every batch len(X) = 64
        loss = 0
        for i in range(len(X)):
            # every image in a batch
            loss += train_(X[i], y[i], encoder, decoder, loss_fn, optimizer_func)
        batch_loss.append(loss)
        if verbose:
            if idx % 100 == 0:
                print(f'Avg loss: {loss} [{idx}/{total}]  {(idx / total) * 100}%')
    return batch_loss


def train_(img, target, encoder, decoder, loss_fn, optim_fn):  # train single image
    optim_fn.zero_grad()

    encoder_hidden = encoder.init_hidden()
    encoder_output, encoder_hidden = encoder(img.unsqueeze(dim=1), encoder_hidden)

    decoder_hidden = encoder_hidden
    encoded_target = lang.indexEncoding(target)
    decoder_input = lang.indexEncoding(torch.Tensor([0]).long())
    decoder_outputs = torch.Tensor()

    use_teacher_forcing = True if random.random() < 0.5 else False

    if use_teacher_forcing:
        for i in range(len(target)):
            output, decoder_hidden = decoder(decoder_input.view(1, 1, -1), decoder_hidden)
            decoder_outputs = torch.cat((decoder_outputs, output.squeeze(1)), 0)
            decoder_input = encoded_target[i]
    else:
        for i in range(len(target)):
            output, decoder_hidden = decoder(decoder_input.view(1, 1, -1), decoder_hidden)
            decoder_outputs = torch.cat((decoder_outputs, output.squeeze(1)), 0)
            decoder_input = output
            topv, topk = output.topk(1)
            if topk.item() == 1: break  # 1 == EOS_token

    try:
        loss = loss_fn(decoder_outputs, target)
        loss.backward()
        optim_fn.step()
        # print(loss.item(), len(target))
        return loss.item() / len(target)

    except:
        traceback.print_exc()
        print(use_teacher_forcing, decoder_outputs.shape, target.shape)
        return 0
