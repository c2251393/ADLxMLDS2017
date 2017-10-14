import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import glob
from util import *
from timit import *
import random
import model_dnn
import model_dcnn

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('feat', default='mfcc',
                    help='mfcc or fbank')
parser.add_argument('model', default='dnn',
                    help='model (dnn or dcnn)')
parser.add_argument('model_file', default='rnn.pt',
                    help='model file')
parser.add_argument('--hidden_size', type=int, default=int(30))
parser.add_argument('--n_layers', type=int, default=int(2))
parser.add_argument('--batch_size', type=int, default=int(32))
parser.add_argument('--frame_size', type=int, default=int(17))
parser.add_argument('--window_size_x', type=int, default=int(3))
parser.add_argument('--window_size_y', type=int, default=int(2))
args = parser.parse_args()


HIDDEN_SIZE = args.hidden_size
FRAME_SIZE = args.frame_size
WINDOW_SIZE = (args.window_size_x, args.window_size_y)
BATCH_SIZE = args.batch_size
N_LAYERS = args.n_layers
DROPOUT = 0

print(args.model, args.model_file, FRAME_SIZE, HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, WINDOW_SIZE, DROPOUT)

timit = TIMIT(args.data, "te", args.feat)

if args.model == "dnn":
    model = model_dnn.DNN(timit.N_FEAT, FRAME_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, 0)
elif args.model == "dcnn":
    model = model_dcnn.CNN(timit.N_FEAT, FRAME_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, 0)

state_dict = torch.load(os.path.join("models", args.model_file), map_location=lambda storage, location: storage)
model.load_state_dict(state_dict)
model = model.eval()

if USE_CUDA:
    model.cuda()

model.eval()

def batch_pre(inp, useful):
    # inp: (BATCH_SIZE x timit.N_FEAT)
    output = model(inp)
    output = output.max(1)[1].data[:useful]
    return list(output)


phseq = {}


for i in range(0, len(timit.te_idx_set), BATCH_SIZE):
    xss, ids, useful = timit.get_flat_batch(i, BATCH_SIZE, FRAME_SIZE, "te")
    print(ids)
    X = Variable(torch.Tensor(xss))
    res = batch_pre(X, useful)
    for (y, (i, j)) in zip(res, ids):
        if i not in phseq:
            phseq[i] = []
        phseq[i].append((j, y))


f = open("submissions/submit_%s.csv" % args.model_file, "w")
f.write("id,phone_sequence\n")

def trim(ys):
    res = []
    pre = -1
    for y in ys:
        if pre != y:
            res.append(y)
        pre = y
    if len(res) == 1:
        return res
    if res[0] == timit.lab2id['sil']:
        res = res[1:]
    if res[-1] == timit.lab2id['sil']:
        res = res[:-1]
    return res

for (id, ys) in phseq.items():
    ys.sort()
    res = [y for (j, y) in ys]
    res = trim(res)
    f.write("%s,%s\n" % (id, ''.join(timit.id2ascii[y] for y in res)))

f.close()

