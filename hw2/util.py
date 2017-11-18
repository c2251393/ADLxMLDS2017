import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import argparse
import os
import json
import glob
import random
import time
import math
import numpy as np
from collections import Counter


MAXLEN = 45
MAX_N_CAP = 40
N_SAMPLE_CAP = 5
VOCAB = 3000
USE_CUDA = torch.cuda.is_available()
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


def parse(ss):
    ss = ss.lower()
    ss = ss.replace(',', '')
    ss = ss.replace('.', '')
    ss = ss.replace('\'', '')
    ss = ss.replace('"', '')
    ss = ss.replace('?', '')
    ss = ss.replace('!', '')
    ss = ss.replace('\n', '')
    ss = ss.replace('/', '')
    ss = ss.replace('\\', '')
    return ss.split()


class LANG():
    def __init__(self):
        self.word2id = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.id2word = {0: "PAD", 1: "SOS", "EOS": 2}

    def index_words(self, ss):
        res = []
        for s in parse(ss):
            res.append(self.index_word(s))
        return res

    def index_word(self, s):
        if s in self.word2id:
            return self.word2id[s]
        return self.word2id["LESS"]

    def build(self, dir):
        data = json.load(open(os.path.join(dir, "training_label.json")))
        data.extend(json.load(open(os.path.join(dir, "testing_label.json"))))

        all_words = []
        for x in data:
            caps = x['caption']
            for ss in caps:
                all_words.extend(parse(ss))

        cnter = Counter(all_words)
        all_words = [k for (k, cnt) in cnter.most_common(VOCAB)]
        all_words.append("LESS")
        for w in all_words:
            self.word2id[w] = len(self.word2id)
            self.id2word[self.word2id[w]] = w


    def tran(self, ss, to_len=-1):
        res = self.index_words(ss)
        if to_len == -1:
            return np.array([SOS_TOKEN] + res + [EOS_TOKEN]), len(res)+2
        if to_len < len(res) + 2:
            print("FUCK")
        tmp = [SOS_TOKEN] + res + [EOS_TOKEN] + [PAD_TOKEN] * (to_len - 2 - len(res))
        return np.array(tmp), len(res)+2

    def itran(self, idxs):
        tokens = []
        for idx in idxs:
            if idx in [EOS_TOKEN, PAD_TOKEN]:
                break
            if idx != SOS_TOKEN:
                word = self.id2word[idx]
                if word == "LESS":
                    continue
                if len(tokens) == 0 or word != tokens[-1]:
                    tokens.append(word)
        res = ' '.join(tokens)
        # res = res[0].upper() + res[1:] + '.'
        return res
    
    def one_hot_word(self, s):
        return one_hot(self.index_word(s))

    def one_hot(self, idx):
        res = [0 for _ in range(len(self.word2id))]
        res[idx] = 1
        return np.array(res)

lang = LANG()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

