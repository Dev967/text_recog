
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets.IAM_words import train_dataset
from utils.data.custom_collate_fn import collate_variable_images
from utils.transforms.encoding import n_letters, letterToIndex, lineToTensor, letterToTensor
from utils.training_script import train
from CONF import *
from models.RNN import Network
import matplotlib.pyplot as plt


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

# if __name__ == '__main__':
if __name__ == 'dontrunme':
    sample_images, sample_labels = next(iter(train_dataloader))
    # print("Sample: ", sample_images[0].shape, sample_labels[0])

    # plt.imshow(sample_images[0], cmap="gray")
    # plt.show()

    myNet = Network(128, 128, 10)
    hidden = myNet.init_hidden()
    output = myNet(sample_images.transpose(0, 1))

    criterion = nn.CTCLoss(blank=52)

    optim_fn = torch.optim.SGD(myNet.parameters(), lr=0.9)
    input_lengths = torch.full(size=(batch_size,), fill_value=256, dtype=torch.long)
    target_lengths = torch.full(size=(batch_size,), fill_value=128, dtype=torch.long)

    loss = criterion(output, sample_labels, input_lengths, target_lengths)
    train(myNet, train_dataloader, criterion, optim_fn)



    # for x in sample_images[0]:
    #     o, hidden = myNet(x, hidden)
    #     print(o.shape)


    # temp = ['tttoooo', 'mmyyyyy', 'hheello', 'ccaarrr']
    # outputs = ['to', 'my', 'he', 'ca']
    #
    # inputs = torch.Tensor()
    # for x in temp:
    #     t = lineToTensor(x)
    #     inputs = torch.cat((inputs, t), dim=1)
    #
    # targets = torch.Tensor([
    #     [19, 14, 55, 55],
    #     [12, 24, 55, 55],
    #     [7, 4, 11, 14],
    #     [2, 0, 17, 55],
    # ])
    #
    # input_lengths = torch.Tensor([7, 7, 7, 7]).long()
    # target_lengths = torch.Tensor([2, 2, 4, 3]).long()
    # criterion = nn.CTCLoss(blank=55)
    # loss = criterion(inputs, targets, input_lengths, target_lengths)
    # print(loss.item())
