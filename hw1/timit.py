import torch
import random
from util import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TIMIT():
    def __init__(self, data_folder, type="tr"):
        self.data = data_folder

        self.lab2id, self.id2ascii = make_lab2id(
            self.data + "48phone_char.map", self.data + "/phones/48_39.map")
        print(self.lab2id)
        print(self.id2ascii)

        if type == "tr":
            train_f = self.data + "mfcc/train.ark"
            train_lab_f = self.data + "label/train.lab"

            # [(targets, inputs)]
            pairs = read_data(train_f, train_lab_f, self.lab2id)

            random.shuffle(pairs)
            self.tr_set, self.valid_set = pairs[100:], pairs[:100]
            print("# of training %d" % (len(self.tr_set)))
            print("# of validation %d" % (len(self.valid_set)))
        elif type == "te":
            # TODO
            pass

    def get_batch(self, i, BATCH_SIZE):
        # xss: [[[feat * 39] * seq len] * BATCH]
        # yss: [[label * seqlen] * BATCH]
        xss = [self.tr_set[i+j][1] for j in range(BATCH_SIZE)]
        yss = [self.tr_set[i+j][0] for j in range(BATCH_SIZE)]
        return xss, yss

