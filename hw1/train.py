import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import glob
from util import *
from timit import *
import random
import model_rnn
import model_cnn
import model_brnn
import model_dnn

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('feat', default='mfcc',
                    help='mfcc or fbank')
parser.add_argument('model', default='rnn',
                    help='model (rnn or cnn or brnn or dnn)')
parser.add_argument('--lr', type=float, default=float(0.1))
parser.add_argument('--n_epoch', type=int, default=int(3))
parser.add_argument('--hidden_size', type=int, default=int(20))
parser.add_argument('--n_layers', type=int, default=int(1))
parser.add_argument('--batch_size', type=int, default=int(32))
parser.add_argument('--window_size_x', type=int, default=int(3))
parser.add_argument('--window_size_y', type=int, default=int(2))
parser.add_argument('--dropout', type=float, default=int(0.0))

args = parser.parse_args()



LR = args.lr
N_EPOCH = args.n_epoch
HIDDEN_SIZE = args.hidden_size
WINDOW_SIZE = (args.window_size_x, args.window_size_y)
BATCH_SIZE = args.batch_size
N_LAYERS = args.n_layers
DROPOUT = args.dropout

print_every = 1
plot_every = 10

print(args.model, args.feat, LR, N_EPOCH, HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, WINDOW_SIZE, DROPOUT)

timit = TIMIT(args.data, "tr", args.feat)

if args.model == "rnn":
    model = model_rnn.RNN(timit.N_FEAT, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "cnn":
    model = model_cnn.CNN(timit.N_FEAT, WINDOW_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "brnn":
    model = model_brnn.BRNN(timit.N_FEAT, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)
elif args.model == "dnn":
    model = model_dnn.DNN(timit.N_FEAT, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)

if USE_CUDA:
    model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr = LR)
# criterion = nn.CrossEntropyLoss(timit.label_wt())
criterion = nn.CrossEntropyLoss()

def train(inp, target, useful, lens):
    # inp: (BATCH_SIZE x maxlen x N_FEAT)
    # target: (BATCH_SIZE x maxlen)
    hidden = model.init_hidden()
    model.zero_grad()
    output, hidden = model(inp, hidden, lens)

    loss = 0

    # my_y = []

    for i in range(useful):
        # my_ys = []
        # tar_ys = []
        for j in range(lens[i]):
            # my_ys.append(output[i][j].topk(1)[1].data[0])
            # tar_ys.append(target[i][j].data[0])
            loss += criterion(output[i][j].view(1, -1), target[i][j])
        # if i == 10:
            # print(' '.join(map(str, my_ys)))
            # print(' '.join(map(str, tar_ys)))

    loss.backward()
    opt.step()

    return loss.data[0] / sum(lens[:useful])


def batch_eval(inp, target, useful, lens):
    # inp: (BATCH_SIZE x maxlen x N_FEAT)
    # target: (BATCH_SIZE x maxlen)
    hidden = model.init_hidden()
    output, hidden = model(inp, hidden, lens)

    acc = 0
    loss = 0

    for i in range(useful):
        for j in range(lens[i]):
            loss += criterion(output[i][j].view(1, -1), target[i][j])
            my_y = output[i][j].topk(1)[1].data[0]
            real_y = target[i][j].data[0]
            if my_y == real_y:
                acc += 1

    return loss.data[0], acc


def eval_valid():
    loss = 0
    acc = 0
    v_len = len(timit.valid_set)
    tot_len = 0
    for i in range(0, v_len, BATCH_SIZE):
        input, target, useful = timit.get_batch(i, BATCH_SIZE, "va")
        input, target, lens = make_batch(input, target, timit.N_FEAT)
        tloss, tacc = batch_eval(input, target, useful, lens)
        loss += tloss
        acc  += tacc
        tot_len += sum(lens[:useful])

    loss /= tot_len
    acc /= tot_len

    print("  VALID LOSS %f ACC %f%%" % (loss, acc * 100))

    return loss, acc


start = time.time()
loss_avg = 0
all_losses = []
loss_tot = 0

iter = 1
eval_valid()
for epoch in range(1, N_EPOCH + 1):
    random.shuffle(timit.tr_set)
    for i in range(0, len(timit.tr_set), BATCH_SIZE):
    # for i in range(0, 100, BATCH_SIZE):
        input, target, useful = timit.get_batch(i, BATCH_SIZE)

        input, target, lens = make_batch(input, target, timit.N_FEAT)

        loss = train(input, target, useful, lens)

        loss_avg += loss
        loss_tot += loss

        if iter % print_every == 0:
            print('[%s (%d %d%%) %.4f %.4f]' %
                  (time_since(start), iter, iter / (N_EPOCH * len(timit.tr_set)) * 100, loss, loss_tot / iter))

        if iter % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0

        iter += 1
    eval_valid()
    torch.save(model.state_dict(), args.model + (".e%d.h%d.b%d.l%d.pt" % (epoch, HIDDEN_SIZE, BATCH_SIZE, N_LAYERS)))

print(all_losses)

