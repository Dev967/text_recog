import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Resize, Compose


compose_1 = Compose([
    ToTensor()
])


def mdrnn_image_transform(img):
    img = img.resize((256, 128))
    img = compose_1(img)
    return img.squeeze(dim=0).transpose(0, 1)
