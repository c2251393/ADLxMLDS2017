#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import argparse
import os
import json
import glob
import random
import numpy as np
from util import *
import model

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('model_file', default='s2vt.h512.b16.e20.pt',
                    help='model file')
parser.add_argument('test_o', default='sample_output_testset.txt',
                    help='test set output')
parser.add_argument('peer_o', default='sample_output_peer_review.txt',
                    help='peer review set output')
parser.add_argument('-l', '--lr', type=float, default=float(0.001))
parser.add_argument('-e', '--n_epoch', type=int, default=int(200))
parser.add_argument('-wx', '--window_size_x', type=int, default=int(3))
parser.add_argument('-wy', '--window_size_y', type=int, default=int(2))
parser.add_argument('-p', '--pool_size', type=int, default=int(2))
parser.add_argument('-H', '--hidden_size', type=int, default=int(256))
parser.add_argument('-b', '--batch_size', type=int, default=int(16))
parser.add_argument('-n', '--n_layers', type=int, default=int(1))
parser.add_argument('-d', '--dropout', type=float, default=int(0.0))
parser.add_argument('-M', '--Model', type=str, default='')
parser.add_argument('-a', '--attn', action='store_true')

args = parser.parse_args()


class MSVD_tr(Dataset):
    def __init__(self, dir):
        self.data = json.load(open(os.path.join(dir, "training_label.json")))
        for cap in self.data:
            id = cap["id"]
            fn = os.path.join(dir, "training_data", "feat", id + ".npy")
            cap["feat"] = np.load(fn).astype('float32')

        for x in self.data:
            caps = x['caption']
            lens = []
            for i in range(len(caps)):
                caps[i], tlen = lang.tran(caps[i], MAXLEN)
                lens.append(tlen)
            for i in range(MAX_N_CAP - len(caps)):
                caps.append(np.array([PAD_TOKEN] * MAXLEN))
                lens.append(0)
            x['cap_lens'] = lens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MSVD_peer(Dataset):
    def __init__(self, dir):
        ids = open(os.path.join(dir, "peer_review_id.txt")).readlines()
        self.data = []
        for id in ids:
            id = id.strip()
            fn = os.path.join(dir, "peer_review", "feat", id + ".npy")
            self.data.append({
                "id": id,
                "feat": np.load(fn).astype('float32')
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MSVD_te(Dataset):
    def __init__(self, dir):
        self.data = json.load(open(os.path.join(dir, "testing_label.json")))
        for cap in self.data:
            id = cap["id"]
            fn = os.path.join(dir, "testing_data", "feat", id + ".npy")
            cap["feat"] = np.load(fn).astype('float32')

        for x in self.data:
            caps = x['caption']
            lens = []
            for i in range(len(caps)):
                caps[i], tlen = lang.tran(caps[i], MAXLEN)
                lens.append(tlen)
            for i in range(MAX_N_CAP - len(caps)):
                caps.append(np.array([PAD_TOKEN] * MAXLEN))
                lens.append(0)
            x['cap_lens'] = lens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

tr_data = MSVD_tr(args.data)
tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True)

te_data = MSVD_te(args.data)
te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=True)

if args.peer_o != 'nan':
    peer_data = MSVD_peer(args.data)
    peer_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=True)


model = model.S2S(args.hidden_size, args.dropout, args.attn)
if USE_CUDA:
    model.cuda()

state_dict = torch.load(os.path.join("models", args.model_file), map_location=lambda storage, location: storage)
model.load_state_dict(state_dict)
model.eval()


def test(batch):
    batch_size = len(batch['id'])
    X = Variable(batch['feat'])
    if USE_CUDA:
        X = X.cuda()

    decoder_outs, symbol_outs = model(X, None, Variable(torch.LongTensor([MAXLEN])))

    return symbol_outs

test_ans = {}
peer_ans = {}


def main():
    start = time.time()

    for (i, bat) in enumerate(te_loader, 1):
        symbol_outs = test(bat)
        for (j, id) in enumerate(bat['id']):
            test_ans[id] = lang.itran(symbol_outs.data[j])
        print("%s %d/%d" % (time_since(start), i, len(te_loader)))

    fp = open(args.test_o, 'w')
    for (k, v) in test_ans.items():
        fp.write("%s,%s\n" % (k, v))
    fp.close()

    if args.peer_o == 'nan':
        return

    for (i, bat) in enumerate(peer_loader, 1):
        symbol_outs = test(bat)
        for (j, id) in enumerate(bat['id']):
            peer_ans[id] = lang.itran(symbol_outs.data[j])
        print("%s %d/%d" % (time_since(start), i, len(te_loader)))

    fp = open(args.peer_o, 'w')
    for (k, v) in peer_ans.items():
        fp.write("%s,%s\n" % (k, v))
    fp.close()

main()
