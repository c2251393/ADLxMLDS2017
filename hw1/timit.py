import torch
import random
from util import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import copy

class TIMIT():
    def __init__(self, data_folder, type="tr", feat="mfcc"):
        self.feat = feat
        self.N_FEAT = 39
        if feat == "fbank":
            self.N_FEAT = 69
        # self.N_LABEL = 48 + 1
        self.N_LABEL = 39 + 1
        self.data = data_folder

        self.lab2id, self.id2ascii = make_lab2id(
            os.path.join(self.data, "48phone_char.map"),
            os.path.join(self.data, "phones", "48_39.map"))
        print(self.lab2id)
        print(self.id2ascii)

        if type == "tr":
            train_f = os.path.join(self.data, feat, "train.ark")
            train_lab_f = os.path.join(self.data, "label", "train.lab")

            # [(targets, inputs)]
            pairs = read_data(train_f, train_lab_f, self.lab2id)

            random.shuffle(pairs)
            self.tr_set = pairs[100:]
            self.tr_idx_set = []
            for i in range(len(self.tr_set)):
                ys, _, _ = self.tr_set[i]
                self.tr_idx_set.extend((i, j) for j in range(len(ys)))

            self.valid_set = pairs[:100]
            self.valid_idx_set = []
            for i in range(len(self.valid_set)):
                ys, _, _ = self.valid_set[i]
                self.valid_idx_set.extend((i, j) for j in range(len(ys)))

            print("# of training %d" % (len(self.tr_set)))
            print("# of validation %d" % (len(self.valid_set)))
        elif type == "te":
            test_f = os.path.join(self.data, feat, "test.ark")
            self.te_set = read_data(test_f, None, self.lab2id)
            self.te_idx_set = []
            for i in range(len(self.te_set)):
                ys, _ = self.te_set[i]
                self.te_idx_set.extend((i, j) for j in range(len(ys)))

            print("# of testing %d" % (len(self.te_set)))


    def label_wt(self):
        # res = torch.zeros(self.N_LABEL)
        # for (ys, _, _) in self.tr_set:
            # for y in ys:
                # res[y] += 1
        # for i in range(self.N_LABEL):
            # if res[i] > 0.0:
                # res[i] = 1.0 / ( res[i] )
            # else:
                # res[i] = 1e9
        res = torch.ones(self.N_LABEL)
        res[self.lab2id['sil']] = 0.3
        if USE_CUDA:
            res = res.cuda()
        return res

    def get_batch(self, i, batch_size, type="tr"):
        if type == "tr":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            xss = [copy.deepcopy(p[1]) for p in self.tr_set[i: i+batch_size]]
            xss += [[[0 for _ in range(self.N_FEAT)]] for _ in range(batch_size - len(xss))]

            yss = [copy.deepcopy(p[0]) for p in self.tr_set[i: i+batch_size]]
            yss += [[0] for _ in range(batch_size - len(yss))]

            sz = len(self.tr_set[i: i+batch_size])
            return xss, yss, sz
        elif type == "va":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: [[label * seqlen] * BATCH]
            xss = [copy.deepcopy(p[1]) for p in self.valid_set[i: i+batch_size]]
            xss += [[[0 for _ in range(self.N_FEAT)]] for _ in range(batch_size - len(xss))]

            yss = [copy.deepcopy(p[0]) for p in self.valid_set[i: i+batch_size]]
            yss += [[0] for _ in range(batch_size - len(yss))]

            sz = len(self.valid_set[i: i+batch_size])
            return xss, yss, sz
        elif type == "te":
            # xss: [[[feat * 39] * seq len] * BATCH]
            # yss: ["id" * BATCH]
            xss = [copy.deepcopy(p[0]) for p in self.te_set[i: i+batch_size]]
            xss += [[[0 for _ in range(self.N_FEAT)]] for _ in range(batch_size - len(xss))]

            ids = [copy.deepcopy(p[1]) for p in self.te_set[i: i+batch_size]]
            ids += ["" for _ in range(batch_size - len(ids))]

            sz = len(self.te_set[i: i+batch_size])
            return xss, ids, sz

    def get_frame(self, i, j, frame_size, type="tr"):
        sz = frame_size // 2
        res = []
        if type == "tr":
            res = get_pad(self.tr_set[i][1], j, sz, sz, [0 for _ in range(self.N_FEAT)])
        elif type == "va":
            res = get_pad(self.valid_set[i][1], j, sz, sz, [0 for _ in range(self.N_FEAT)])
        elif type == "te":
            res = get_pad(self.te_set[i][0], j, sz, sz, [0 for _ in range(self.N_FEAT)])
        return res

    def get_flat_batch(self, i, batch_size, frame_size, type="tr"):
        if type == "tr":
            xss = [self.get_frame(p[0], p[1], frame_size, type) for p in self.tr_idx_set[i: i+batch_size]]
            xss += [
                [[0 for _ in range(self.N_FEAT)] for _ in range(frame_size)]
                for _ in range(batch_size-len(xss))
            ]

            yss = [self.tr_set[p[0]][0][p[1]] for p in self.tr_idx_set[i: i+batch_size]]
            yss += [0 for _ in range(batch_size - len(yss))]

            sz = len(self.tr_idx_set[i: i + batch_size])
            return xss, yss, sz
        elif type == "va":
            xss = [self.get_frame(p[0], p[1], frame_size, type) for p in self.valid_idx_set[i: i+batch_size]]
            xss += [
                [[0 for _ in range(self.N_FEAT)] for _ in range(frame_size)]
                for _ in range(batch_size-len(xss))
            ]

            yss = [self.valid_set[p[0]][0][p[1]] for p in self.valid_idx_set[i: i+batch_size]]
            yss += [0 for _ in range(batch_size - len(yss))]

            sz = len(self.valid_idx_set[i: i + batch_size])
            return xss, yss, sz
        elif type == "te":
            xss = [self.get_frame(p[0], p[1], frame_size, type) for p in self.te_idx_set[i: i+batch_size]]
            xss += [
                [[0 for _ in range(self.N_FEAT)] for _ in range(frame_size)]
                for _ in range(batch_size-len(xss))
            ]

            ids = [(self.te_set[p[0]][1], p[1]) for p in self.te_idx_set[i: i+batch_size]]
            ids += [("", -1) for _ in range(batch_size - len(ids))]

            sz = len(self.te_idx_set[i: i + batch_size])
            return xss, ids, sz
