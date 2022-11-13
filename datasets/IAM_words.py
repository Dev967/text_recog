import traceback

from PIL import Image
from torch.utils.data import Dataset, random_split

from CONF import *
from utils.data.lang_handle import Lang
from utils.datasets.IAM.reorder_wordsdir import reorder
from utils.transforms.image_transforms import mdrnn_image_transform


class IAMWords(Dataset):
    def __init__(self, pairs, lang, transform=None, target_transform=None):
        reorder()
        self.transform = transform
        self.target_transform = target_transform
        self.lang = lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_file, target = self.pairs[idx]

        if self.transform: image_file = self.transform()
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

train_dataset, test_dataset = random_split(whole_dataset, [86456, 10000])

del pairs
