import torch
import random
from util import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import copy
import numpy as np

class TIMIT():
    def __init__(self, data_folder, type="tr", feat="mfcc"):
        self.feat = feat
        self.N_FEAT = 39
        if feat == "fbank":
            self.N_FEAT = 69
        elif feat == "all":
            self.N_FEAT = 39 + 69
        self.N_LABEL = 39 + 1
        self.data = data_folder

        self.lab2id, self.id2ascii = make_lab2id(
            os.path.join(self.data, "48phone_char.map"),
            os.path.join(self.data, "phones", "48_39.map"))
        print(self.lab2id)
        print(self.id2ascii)

        if type == "tr":
            train_f = []
            if feat == "all":
                train_f.append(os.path.join(self.data, "mfcc", "train.ark"))
                train_f.append(os.path.join(self.data, "fbank", "train.ark"))
            else:
                train_f.append(os.path.join(self.data, feat, "train.ark"))
            train_lab_f = os.path.join(self.data, "label", "train.lab")

            # [(targets, inputs, id)]
            pairs = read_data(train_f, train_lab_f, self.lab2id)
            self.max_len = max(map(lambda p: len(p[0]), pairs))

            random.shuffle(pairs)

            self.tr_set = pairs[100:]

            self.valid_set = pairs[:100]

            print("# of training %d" % (len(self.tr_set)))
            print("# of validation %d" % (len(self.valid_set)))
        elif type == "te":
            test_f = []
            if feat == "all":
                test_f.append(os.path.join(self.data, "mfcc", "test.ark"))
                test_f.append(os.path.join(self.data, "fbank", "test.ark"))
            else:
                test_f.append(os.path.join(self.data, feat, "test.ark"))

            # [(inputs, id)]
            self.te_set = read_data(test_f, None, self.lab2id)
            self.max_len = max(map(lambda p: len(p[0]), self.te_set))

            print("# of testing %d" % (len(self.te_set)))


    def label_wt(self):
        res = torch.ones(self.N_LABEL)
        res[self.lab2id['sil']] = 0.3
        if USE_CUDA:
            res = res.cuda()
        return res

    def get_batch(self, i, batch_size, type="tr"):
        if type == "tr":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            sz = len(self.tr_set[i: i+batch_size])

            xss = [xs.copy() for (ys, xs, id) in self.tr_set[i: i + batch_size]]
            xss += [np.zeros((1, self.N_FEAT)) for _ in range(batch_size - sz)]

            yss = [ys.copy() for (ys, xs, id) in self.tr_set[i: i + batch_size]]
            yss += [np.zeros(1) for _ in range(batch_size - sz)]

            ids = [id for (ys, xs, id) in self.tr_set[i: i + batch_size]]
            ids += ["" for _ in range(batch_size - sz)]

            return xss, yss, ids, sz
        elif type == "va":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            sz = len(self.valid_set[i: i+batch_size])

            xss = [xs.copy() for (ys, xs, id) in self.valid_set[i: i + batch_size]]
            xss += [np.zeros((1, self.N_FEAT)) for _ in range(batch_size - sz)]

            yss = [ys.copy() for (ys, xs, id) in self.valid_set[i: i + batch_size]]
            yss += [np.zeros(1) for _ in range(batch_size - sz)]

            ids = [id for (ys, xs, id) in self.valid_set[i: i + batch_size]]
            ids += ["" for _ in range(batch_size - sz)]

            return xss, yss, ids, sz
        elif type == "te":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: ["id" * BATCH]
            sz = len(self.te_set[i: i+batch_size])

            xss = [xs.copy() for (xs, id) in self.te_set[i: i + batch_size]]
            xss += [np.zeros((1, self.N_FEAT)) for _ in range(batch_size - sz)]

            ids = [id for (xs, id) in self.te_set[i: i + batch_size]]
            ids += ["" for _ in range(batch_size - sz)]

            return xss, ids, sz
