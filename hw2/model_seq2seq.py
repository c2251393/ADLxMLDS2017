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
import math
import numpy as np
from util import *
import model

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
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
parser.add_argument('-s', '--sample', type=str, default='ref')

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

model = model.S2S(args.hidden_size, EMBED_SIZE, args.dropout, args.attn)
if USE_CUDA:
    model.cuda()

opt = torch.optim.Adam(model.parameters(), lr = args.lr)
criterion = nn.CrossEntropyLoss()


def train(batch, sched_sampling_p=1):
    model.train()
    opt.zero_grad()
    loss = 0

    batch_size = len(batch['id'])
    cap_idxs = []
    for i in range(batch_size):
        good_idxs = [j for j in range(MAX_N_CAP) if batch['cap_lens'][j][i] > 0]
        cap_idxs.append(random.sample(good_idxs, 5))

    X = Variable(batch['feat'])
    target_outputs = Variable(torch.stack([
        batch['caption'][cids[0]][i] for (i, cids) in enumerate(cap_idxs)]))
    target_lengths = Variable(torch.LongTensor([
        batch['cap_lens'][cids[0]][i] for (i, cids) in enumerate(cap_idxs)]))
    if USE_CUDA:
        X = X.cuda()
        target_outputs = target_outputs.cuda()
        target_lengths = target_lengths.cuda()

    decoder_outs, symbol_outs = model(X, target_outputs, target_lengths, sched_sampling_p)

    tot_len = 0

    for i in range(batch_size):
        cids = cap_idxs[i]
        for j in range(len(cids)):
            tlen = batch['cap_lens'][cids[j]][i]
            tot_len += tlen
            sent = Variable(batch['caption'][cids[j]][i][:tlen])
            if USE_CUDA:
                sent = sent.cuda()
            loss += criterion(decoder_outs[i][:tlen], sent)

    if USE_CUDA:
        loss = loss.cuda()
    loss.backward()
    opt.step()

    return loss.data[0] / tot_len


def eval(batch):
    model.eval()

    batch_size = len(batch['id'])

    X = Variable(batch['feat'])
    cap_idxs = [[] for i in range(batch_size)]
    for i in range(batch_size):
        good_idxs = [j for j in range(MAX_N_CAP) if batch['cap_lens'][j][i] > 0]
        cap_idxs[i] = random.sample(good_idxs, 5)

    target_outputs = Variable(torch.stack([
        batch['caption'][cids[0]][i] for (i, cids) in enumerate(cap_idxs)]))
    target_lengths = Variable(torch.LongTensor([
        batch['cap_lens'][cids[0]][i] for (i, cids) in enumerate(cap_idxs)]))
    if USE_CUDA:
        X = X.cuda()
        target_outputs = target_outputs.cuda()
        target_lengths = target_lengths.cuda()

    decoder_outs, symbol_outs = model(X, None, Variable(torch.LongTensor([MAXLEN])))

    loss = 0

    tot_len = 0

    for i in range(batch_size):
        cids = cap_idxs[i]
        for j in range(len(cids)):
            tlen = batch['cap_lens'][cids[j]][i]
            tot_len += tlen
            sent = Variable(batch['caption'][cids[j]][i][:tlen])
            if USE_CUDA:
                sent.cuda()
            loss += criterion(decoder_outs[i][:tlen], sent)

    return loss.data[0] / tot_len, symbol_outs

test_ans = {}

def main():
    start = time.time()
    iter = 1
    for epoch in range(1, args.n_epoch+1):
        print("================= EPOCH %d ======================" % epoch)
        if args.sample == 'ref':
            prob = 1
        elif args.sample == 'sched':
            prob = 1.0 - epoch / args.n_epoch
            # k = 64
            # prob = k / (k + math.exp(iter / k))
        for (i, bat) in enumerate(tr_loader, 1):
            loss = train(bat, prob)
            if i % 1 == 0:
                print("%s %d/%d %.4f p=%.2f" % (time_since(start), i, len(tr_loader), loss, prob))
            iter += 1

        for (i, bat) in enumerate(te_loader, 1):
            loss, symbol_outs = eval(bat)
            for (j, id) in enumerate(bat['id']):
                test_ans[id] = lang.itran(symbol_outs.data[j])

            print("%s %d/%d %.4f" % (time_since(start), i, len(te_loader), loss))

        model_name = "s2vt.h%d.b%d.e%d.%s.pt" % (args.hidden_size, args.batch_size, epoch, args.sample)

        if epoch % 1 == 0:
            fp = open(model_name + ".ans", 'w')
            for (k, v) in test_ans.items():
                fp.write("%s,%s\n" % (k, v))
            fp.close()
            torch.save(model.state_dict(), os.path.join("models", model_name))

# main()
