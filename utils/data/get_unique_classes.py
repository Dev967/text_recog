import torch


def makeUniqueTensor(sample_imgs, sample_labels):
    classes = torch.Tensor([])

    for x in range(10):
        idx = (sample_labels == x).nonzero(as_tuple=True)[0][0]
        classes = torch.cat((classes, sample_imgs[idx]), 0)
    return classes
