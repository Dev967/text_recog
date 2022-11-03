from torchvision.datasets import MNIST
import torchvision.transforms as T
from CONF import MNIST_path

train_ds = MNIST(root=MNIST_path, download=True, train=True, transform=T.ToTensor())
test_ds = MNIST(root=MNIST_path, download=True, train=False, transform=T.ToTensor())
