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
parser.add_argument('-b', '--batch_size', type=int, default=int(16))
parser.add_argument('-B', '--beam_search', type=int, default=int(-1))
parser.add_argument('-x', '--special', action='store_true')

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

lang.build(args.data)

tr_data = MSVD_tr(args.data)
tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True)

te_data = MSVD_te(args.data)
if args.beam_search == -1:
    te_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=True)
else:
    te_loader = DataLoader(te_data, batch_size=1, shuffle=True)

if args.peer_o != 'nan':
    peer_data = MSVD_peer(args.data)
    if args.beam_search == -1:
        peer_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=True)
    else:
        peer_loader = DataLoader(te_data, batch_size=1, shuffle=True)


state_dict = torch.load(os.path.join("models", args.model_file), map_location=lambda storage, location: storage)
hidden_size = state_dict["hidden_size"]
embed_size = state_dict["embed_size"]
n_layers = state_dict["n_layers"]
attn = state_dict["attn"]

encoder = model.Encoder(hidden_size, n_layers, 0)
decoder = model.Decoder(hidden_size, embed_size, n_layers, 0, attn)
encoder.load_state_dict(state_dict["encoder"])
decoder.load_state_dict(state_dict["decoder"])
encoder.eval()
decoder.eval()
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

print("beam search ", args.beam_search)

def test(batch):
    batch_size = len(batch['id'])
    X = Variable(batch['feat'])
    if USE_CUDA:
        X = X.cuda()

    encoder_outputs, hidden = encoder(X)

    decoder_input = Variable(torch.LongTensor([SOS_TOKEN] * batch_size))
    decoder_context = Variable(torch.zeros(batch_size, 1, args.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    output_symbols = []

    for i in range(1, MAXLEN):
        decoder_output, decoder_input, hidden, decoder_context = decoder(
            decoder_input, hidden, decoder_context, encoder_outputs)
        output_symbols.append(decoder_input)

    output_symbols = torch.stack(output_symbols, 1)

    return output_symbols

test_ans = {}
peer_ans = {}

special = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']

def main():
    start = time.time()
    print(args.special)

    for (i, bat) in enumerate(te_loader, 1):
        symbol_outs = test(bat)
        for (j, id) in enumerate(bat['id']):
            test_ans[id] = lang.itran(symbol_outs.data[j])
        print("%s %d/%d" % (time_since(start), i, len(te_loader)))

    fp = open(args.test_o, 'w')
    for (k, v) in test_ans.items():
        if not args.special or k in special:
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
