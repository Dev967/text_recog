import torch


class Lang:
    def __init__(self, pairs):
        self.char_classes = ""
        self.charToIndex = {0: "SOS", 1: "EOS"}
        self.indexToCharacter = {}
        self.n_chars = 2

        for file, word in pairs:
            for char in word:
                if char not in self.char_classes:
                    self.char_classes += char
                    self.charToIndex[char] = self.n_chars
                    self.indexToCharacter[self.n_chars] = char
                    self.n_chars += 1

    def wordToIndex(self, word):
        output = []
        for char in word:
            output.append(self.charToIndex[char])
        return torch.Tensor(output).long()

    def indexEncoding(self, tensor):
        output_tensor = torch.Tensor()
        for i in tensor:
            temp = torch.zeros(self.n_chars)
            temp[i] = 1
            output_tensor = torch.cat((output_tensor, temp.unsqueeze(dim=0)), dim=0)
        return output_tensor
