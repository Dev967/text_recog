import torch


def simple_vertical_segmentation(img):
    arr = torch.sum(img, dim=0)
    return arr

