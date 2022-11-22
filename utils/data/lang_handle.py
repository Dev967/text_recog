import os
import pickle

import torch


class Lang:
    def __init__(self, pairs, use_cache):
        self.char_classes = ""
        self.charToIndex = {0: "SOS", 1: "EOS"}
        self.indexToCharacter = {}
        self.n_chars = 2

        self.curr_dir = 'cache/lang'

        if use_cache:
            # using cache
            print("> using cache for lang")

            file_char_classes = open(f'{self.curr_dir}/charClasses.txt', 'r')
            file_charToIndex = open(f'{self.curr_dir}/charToIndex.txt', 'rb')
            file_indexToCharacter = open(f'{self.curr_dir}/indexToCharacter.txt', 'rb')
            file_n_chars = open(f'{self.curr_dir}/n_chars.txt', 'r')

            self.char_classes = file_char_classes.read()
            line = file_n_chars.read()
            self.n_chars = int(line)

            bytes_charToIndex = file_charToIndex.read()
            bytes_indexToCharacter = file_indexToCharacter.read()

            self.charToIndex = pickle.loads(bytes_charToIndex)
            self.indexToCharacter = pickle.loads(bytes_indexToCharacter)

        else:
            os.makedirs(self.curr_dir, exist_ok=True)

            file_char_classes = open(f'{self.curr_dir}/charClasses.txt', 'w')
            file_charToIndex = open(f'{self.curr_dir}/charToIndex.txt', 'wb')
            file_indexToCharacter = open(f'{self.curr_dir}/indexToCharacter.txt', 'wb')
            file_n_chars = open(f'{self.curr_dir}/n_chars.txt', 'w')

            for file, word in pairs:
                for char in word:
                    if char not in self.char_classes:
                        self.char_classes += char
                        self.charToIndex[char] = self.n_chars
                        self.indexToCharacter[self.n_chars] = char
                        self.n_chars += 1

            # Caching
            file_char_classes.write(self.char_classes)

            pickle.dump(self.charToIndex, file_charToIndex)
            pickle.dump(self.indexToCharacter, file_indexToCharacter)

            file_n_chars.write(str(self.n_chars))

        file_char_classes.close()
        file_charToIndex.close()
        file_indexToCharacter.close()
        file_n_chars.close()

    def wordToIndex(self, word):
        # turns word to index, str -> tensor
        output = []
        for char in word:
            output.append(self.charToIndex[char])
        return torch.Tensor(output).long()

    def indexEncoding(self, tensor):
        # turns integer to one hot encoding, int -> tensor
        output_tensor = torch.Tensor()
        for i in tensor:
            temp = torch.zeros(self.n_chars)
            temp[i] = 1
            output_tensor = torch.cat((output_tensor, temp.unsqueeze(dim=0)), dim=0)
        return output_tensor

    def tensorToWord(self, tensor):
        # turns integer to word, int -> word
        output = ""
        for i in tensor:
            output += self.indexToCharacter[i.item()]
        return output
