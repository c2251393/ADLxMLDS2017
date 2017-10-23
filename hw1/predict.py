import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import glob
from util import *
from timit import *
import random
import model_rnn
import model_cnn
import model_brnn
import model_bcnn
import model_res

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('feat', default='mfcc',
                    help='mfcc or fbank')
parser.add_argument('model', default='rnn',
                    help='model (rnn or cnn or brnn or bcnn or res)')
parser.add_argument('model_file', default='rnn.pt',
                    help='model file')
parser.add_argument('-wx', '--window_size_x', type=int, default=int(3))
parser.add_argument('-wy', '--window_size_y', type=int, default=int(2))
parser.add_argument('-p', '--pool_size', type=int, default=int(2))
parser.add_argument('-H', '--hidden_size', type=int, default=int(20))
parser.add_argument('-b', '--batch_size', type=int, default=int(32))
parser.add_argument('-n', '--n_layers', type=int, default=int(1))
parser.add_argument('-d', '--dropout', type=float, default=int(0.0))
args = parser.parse_args()


HIDDEN_SIZE = args.hidden_size
WINDOW_SIZE = (args.window_size_x, args.window_size_y)
POOL_SIZE = args.pool_size
BATCH_SIZE = args.batch_size
N_LAYERS = args.n_layers
DROPOUT = args.dropout

print(args.model, args.model_file, HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, WINDOW_SIZE, DROPOUT)

timit = TIMIT(args.data, "te", args.feat)

if args.model == "rnn":
    model = model_rnn.RNN(timit.N_FEAT, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "brnn":
    model = model_brnn.BRNN(timit.N_FEAT, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "cnn":
    model = model_cnn.CNN(timit.N_FEAT, WINDOW_SIZE, POOL_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "bcnn":
    model = model_bcnn.BCNN(timit.N_FEAT, WINDOW_SIZE, POOL_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "res":
    model = model_res.RESR(timit.N_FEAT, timit.N_LABEL, BATCH_SIZE, DROPOUT)

state_dict = torch.load(os.path.join("models", args.model_file), map_location=lambda storage, location: storage)
model.load_state_dict(state_dict)
model = model.eval()

if USE_CUDA:
    model.cuda()


def batch_pre(inp, useful, lens):
    # inp: (BATCH_SIZE x maxlen x timit.N_FEAT)
    # target: (BATCH_SIZE x maxlen)
    hidden = model.init_hidden()
    output, hidden = model(inp, hidden, lens)
    res = []

    for i in range(useful):
        ys = []
        for j in range(lens[i]):
            y = output[i][j].topk(1)[1].data[0]
            ys.append(y)
        res.append(ys)

    return res

f = open("submissions/submit_%s.csv" % args.model_file, "w")
f.write("id,phone_sequence\n")

def trim(ys):
    res = []
    pre = -1
    cnt = 0
    for y in ys:
        if pre != y:
            if cnt > 2:
                res.append(pre)
            pre = y
            cnt = 1
        else:
            cnt += 1
    if cnt > 2:
        res.append(pre)
    if len(res) == 1:
        return res
    if res[0] == timit.id2ascii[timit.lab2id['sil']]:
        res = res[1:]
    if res[-1] == timit.id2ascii[timit.lab2id['sil']]:
        res = res[:-1]
    return res

for i in range(0, len(timit.te_set), BATCH_SIZE):
    inputs, ids, useful = timit.get_batch(i, BATCH_SIZE, "te")
    inputs, ids, lens = make_batch_te(inputs, ids, timit.N_FEAT)
    # print(ids)
    res = batch_pre(inputs, useful, lens)
    for j in range(useful):
        ans = ''.join(timit.id2ascii[y] for y in res[j])
        ans = ''.join(trim(ans))
        f.write("%s,%s\n" % (ids[j], ans))

f.close()

