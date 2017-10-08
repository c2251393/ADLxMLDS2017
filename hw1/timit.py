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
            test_f = self.data + "mfcc/test.ark"
            self.te_set = read_data(test_f, None, self.lab2id)
            print("# of testing %d" % (len(self.te_set)))


    def get_batch(self, i, batch_size, type="tr"):
        if type == "tr":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            xss = [p[1] for p in self.tr_set[i: i+batch_size]]
            xss += [[[0 for _ in range(N_FEAT)]] for _ in range(batch_size - len(xss))]
            yss = [p[0] for p in self.tr_set[i: i+batch_size]]
            yss += [[0] for _ in range(batch_size - len(yss))]
            sz = len(self.tr_set[i: i+batch_size])
            return xss, yss, sz
        elif type == "va":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            xss = [p[1] for p in self.valid_set[i: i+batch_size]]
            xss += [[[0 for _ in range(N_FEAT)]] for _ in range(batch_size - len(xss))]
            yss = [p[0] for p in self.valid_set[i: i+batch_size]]
            yss += [[0] for _ in range(batch_size - len(yss))]
            sz = len(self.valid_set[i: i+batch_size])
            return xss, yss, sz
        elif type == "te":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: ["id" * BATCH]
            xss = [p[0] for p in self.te_set[i: i+batch_size]]
            xss += [[[0 for _ in range(N_FEAT)]] for _ in range(batch_size - len(xss))]
            ids = [p[1] for p in self.te_set[i: i+batch_size]]
            ids += ["" for _ in range(batch_size - len(ids))]
            sz = len(self.te_set[i: i+batch_size])
            return xss, ids, sz
