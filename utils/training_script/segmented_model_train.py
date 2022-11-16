import random
import traceback

import torch

from CONF import *
from datasets.IAM_words import lang


# functions for training individual images
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
        if not supress_errors: traceback.print_exc()
        print(use_teacher_forcing, decoder_outputs.shape, target.shape)
        return 0


def train_attention(img, target, encoder, decoder, loss_fn, optim_fn):
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
            output, decoder_hidden, attn_weights = decoder(decoder_input.view(1, 1, -1), decoder_hidden, encoder_output)
            decoder_outputs = torch.cat((decoder_outputs, output.squeeze(1)), 0)
            decoder_input = encoded_target[i]
    else:
        for i in range(len(target)):
            output, decoder_hidden, attn_weights = decoder(decoder_input.view(1, 1, -1), decoder_hidden, encoder_output)
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
        if not supress_errors: traceback.print_exc()
        print(use_teacher_forcing, decoder_outputs.shape, target.shape)
        return 0


def train(encoder, decoder, dataloader, loss_fn, optimizer_func, verbose=True, attention_enabled=False):
    batch_loss = []
    total = len(dataloader.dataset)
    count = 0
    batch = 0

    if attention_enabled:
        # use attention training function
        for _, (X, y) in enumerate(dataloader):

            batch += 1
            # every batch len(X) = 64

            loss = 0
            for i in range(len(X)):
                count += 1
                # every image in a batch

                loss += train_attention(X[i], y[i], encoder, decoder, loss_fn, optimizer_func)

            batch_loss.append(loss)
            if verbose:
                if batch % 100 == 0:
                    print(
                        f'Avg loss: {sum(batch_loss) / batch} most recent loss: {loss} [{count}/{total}]  {(count / total) * 100}%')

        return batch_loss

    else:
        # use normal training function
        for _, (X, y) in enumerate(dataloader):
            batch += 1
            # every batch len(X) = 64

            loss = 0
            for i in range(len(X)):
                count += 1
                # every image in a batch

                loss += train_(X[i], y[i], encoder, decoder, loss_fn, optimizer_func)

            batch_loss.append(loss)
            if verbose:
                if batch % 100 == 0:
                    print(f'Avg loss: {loss} [{count}/{total}]  {(count / total) * 100}%')
        return batch_loss
