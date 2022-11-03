import torch
from CONF import *
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class IIT5K_train(Dataset):
    def __init__(self, dir, transform, target_transform, train=True, use_cache=False):
        self.use_cache = use_cache
        self.cached_data = []
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.dir = dir
        self.target = sio.loadmat(f'{dir}/traindata.mat')['traindata'][0] if train else \
        sio.loadmat(f'{dir}/testdata.mat')['testdata'][0]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if self.use_cache:
            return self.cached_data[idx]
        else:
            t = self.target[idx]
            target = t[1][0]
            print("PATH: ", t[0][0])
            image = Image.open(f'{self.dir}/{t[0][0]}')
            if self.transform: image = self.transform(image)
            if self.target_transform: target = self.target_transform(target)

            # cache
            self.cached_data.append([image, target])

            return [image, target]

    def set_use_cache(self, use_cache):
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)
        else:
            self.cached_data = []
        self.use_cache = use_cache


train_dataset = IIT5K_train(IIT5K_path, ToTensor(), None, train=True, use_cache=False)
