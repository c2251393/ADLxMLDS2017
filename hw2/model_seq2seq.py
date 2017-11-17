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
parser.add_argument('-l', '--lr', type=float, default=float(0.0001))
parser.add_argument('-e', '--n_epoch', type=int, default=int(300))
parser.add_argument('-H', '--hidden_size', type=int, default=int(256))
parser.add_argument('-E', '--embed_size', type=int, default=int(256))
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

lang.build(args.data)

tr_data = MSVD_tr(args.data)
tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True)

te_data = MSVD_te(args.data)
te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=True)

encoder = model.Encoder(args.hidden_size, args.n_layers, args.dropout)
decoder = model.Decoder(args.hidden_size, args.embed_size, args.n_layers, args.dropout, args.attn)
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

encoder_opt = torch.optim.Adam(encoder.parameters(), lr=args.lr)
decoder_opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()


def do_batch(batch, sched_sampling_p=1, train=True):
    batch_size = len(batch['id'])
    caption = batch['caption']
    # caption: N_CAP * (batch, MAXLEN)
    cap_lens = batch['cap_lens']
    # cap_lens: N_CAP * (batch)
    cap_idxs = []
    for i in range(batch_size):
        good_idxs = [j for j in range(MAX_N_CAP) if cap_lens[j][i] > 0]
        cap_idxs.append(random.sample(good_idxs, N_SAMPLE_CAP))

    X = Variable(batch['feat'])

    target_outputs = []
    target_lengths = []
    for j in range(N_SAMPLE_CAP):
        tmp = []
        tmp_lens = []
        for i in range(batch_size):
            tmp.append(caption[cap_idxs[i][j]][i])
            tmp_lens.append(cap_lens[cap_idxs[i][j]][i])
        tmp = torch.stack(tmp)
        # tmp: (batch, MAXLEN)
        target_outputs.append(tmp)
        target_lengths.append(tmp_lens)
    target_outputs = Variable(torch.stack(target_outputs))
    # target_outputs: (N_CAP, batch, MAXLEN)
    # target_length: [N_CAP, batch]

    if USE_CUDA:
        X = X.cuda()
        target_outputs = target_outputs.cuda()

    if train:
        encoder.train()
        decoder.train()
        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
    else:
        encoder.eval()
        decoder.eval()

    encoder_outputs, hidden = encoder(X)

    use_teacher = train and random.random() < sched_sampling_p

    decoder_input = target_outputs[0, :, 0]
    decoder_context = Variable(torch.zeros(batch_size, 1, args.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    loss = 0
    tot_len = 0
    for j in range(N_SAMPLE_CAP):
        for i in range(batch_size):
            tlen = target_lengths[j][i]
            tot_len += tlen

    decoder_outputs = []
    output_symbols = []

    for i in range(1, MAXLEN):
        decoder_output, decoder_input, hidden, decoder_context = decoder(
            decoder_input, hidden, decoder_context, encoder_outputs)

        decoder_outputs.append(decoder_output)
        output_symbols.append(decoder_input)

        if use_teacher:
            decoder_input = target_outputs[0, :, i]

    decoder_outputs = torch.stack(decoder_outputs, 1) #(batch, MAXLEN, VOCAB)
    output_symbols = torch.stack(output_symbols, 1) #(batch, MAXLEN)

    for j in range(N_SAMPLE_CAP):
        for i in range(batch_size):
            tlen = target_lengths[j][i]
            loss += criterion(decoder_outputs[i][:tlen], target_outputs[j][i][:tlen])

    if train:
        if USE_CUDA:
            loss = loss.cuda()
        loss.backward()
        encoder_opt.step()
        decoder_opt.step()

    return loss.data[0] / tot_len, output_symbols

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
            loss, _ = do_batch(bat, prob, True)
            if i % 30 == 0:
                print("%s %d/%d %.4f p=%.2f" % (time_since(start), i, len(tr_loader), loss, prob))
            iter += 1

        for (i, bat) in enumerate(te_loader, 1):
            loss, symbol_outs = do_batch(bat, 1, False)
            for (j, id) in enumerate(bat['id']):
                test_ans[id] = lang.itran(symbol_outs.data[j])

            print("%s %d/%d %.4f" % (time_since(start), i, len(te_loader), loss))

        model_name = "s2vt.H%d.E%d.N%d.A%d.b%d.e%d.%s.pt" % (args.hidden_size,
                                                             args.embed_size,
                                                             args.n_layers,
                                                             1 if args.attn else 0,
                                                             args.batch_size,
                                                             epoch,
                                                             args.sample)

        fp = open(model_name + ".ans", 'w')
        for (k, v) in test_ans.items():
            fp.write("%s,%s\n" % (k, v))
        fp.close()
        if epoch % 20 == 0:
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "hidden_size": args.hidden_size,
                    "embed_size": args.embed_size,
                    "n_layers": args.n_layers,
                    "attn": args.attn
                },
                os.path.join("models", model_name)
            )

main()
