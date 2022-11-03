import unicodedata
import string
import torch

all_letters = string.ascii_letters + " .,;'\"1234567890!@#$%^&*()[]{}?+/='<>"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def letterToIndex(letter):
    return all_letters.find(letter)


# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def lineToIndex(line):
    arr = torch.full(size=(128,), fill_value=52, dtype=torch.long)
    for li, letter in enumerate(line):
        idx = letterToIndex(letter)
        if idx >= 256: idx = 52
        try:
            arr[li] = idx
        except:
            raise "HELL"
    return arr
