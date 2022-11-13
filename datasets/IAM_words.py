import math
import traceback

from PIL import Image
from torch.utils.data import Dataset, random_split

from CONF import *
from utils.data.lang_handle import Lang
from utils.transforms.image_transforms import mdrnn_image_transform


class IAMWords(Dataset):
    def __init__(self, pairs, lang, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.lang = lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_file, target = self.pairs[idx]
        image_file = Image.open(image_file)

        if self.transform: image_file = self.transform(image_file)
        if self.target_transform: target = self.target_transform(target)

        return [image_file, target]


file = open(target_file)
pairs = []
for line in file.readlines():
    arr = line.strip().split(" ")
    word = ""

    if arr[1] != "ok": continue

    for i in range(8, len(arr)):
        word += arr[i]

    try:
        image = Image.open(f'{image_dir}/{arr[0]}.png')
        pair = [f'{image_dir}/{arr[0]}.png', word]
        pairs.append(pair)

    except:
        print("failed to read image ", f'{image_dir}/{arr[0]}.png')
        traceback.print_exc()

lang = Lang(pairs)

whole_dataset = IAMWords(pairs, lang, transform=mdrnn_image_transform, target_transform=lang.wordToIndex)
train_size = math.floor(len(whole_dataset) * 80 / 100)
test_size = math.floor(len(whole_dataset) * 20 / 100)
print(f'Train/Test dataset Size: {train_size}/{test_size} whole dataset size: {len(whole_dataset)}')
if train_size + test_size < len(whole_dataset):
    train_size += len(whole_dataset) - (train_size + test_size)

train_dataset, test_dataset = random_split(whole_dataset, [train_size, test_size])

del pairs
