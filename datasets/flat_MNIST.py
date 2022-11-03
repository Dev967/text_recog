from torchvision.datasets import MNIST
from utils.transforms.image import image_to_tensor

train_ds = MNIST(root='data', download=True, train=True, transform=image_to_tensor)
test_ds = MNIST(root='data', download=True, train=False, transform=image_to_tensor)
