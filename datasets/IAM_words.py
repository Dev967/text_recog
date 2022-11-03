import pandas as pd
from torch.utils.data import Dataset, random_split
from CONF import *
from PIL import Image
from utils.transforms.image_transforms import mdrnn_image_transform
from utils.transforms.encoding import *

columns = ['filename', 'status', 'grey level', 'x', 'y', 'w', 'h', 'grammatical tag', 'word']


def bad_line_handler(x):
    size = len(x)
    word = ''
    if size >= 10:
        for i in range(8, size):
            word += f'{x.pop(8)} '
        return x.append(word)
    else:
        raise Exception(f'Not size problem... size={size}')


dataframe = pd.read_csv(target_file, delimiter=' ', engine="python", on_bad_lines=bad_line_handler)
dataframe.columns = columns
pd.set_option("display.max_columns", None)


class IAMWords(Dataset):
    def __init__(self, data_dir, df, transform=None, target_transform=None):
        self.dataset_dir = data_dir
        self.target = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        item = self.target.iloc[idx]
        file = Image.open(f'{dataset_dir}/{item.filename}.png')
        target = item.word
        if self.transform: file = self.transform(file)
        try:
            if self.target_transform: target = self.target_transform(item.word)
        except:
            print("THE WORD: ", item.word)
        return [file, target]


dataframe = dataframe[dataframe.status == 'ok']

whole_dataset = IAMWords(dataset_dir, dataframe, transform=mdrnn_image_transform, target_transform=lineToIndex)
train_dataset, test_dataset = random_split(whole_dataset, [63968, 10000])
