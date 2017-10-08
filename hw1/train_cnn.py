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

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('--lr', type=float, default=float(0.1))
parser.add_argument('--n_epoch', type=int, default=int(3))
parser.add_argument('--hidden_size', type=int, default=int(20))
parser.add_argument('--n_layers', type=int, default=int(1))
parser.add_argument('--batch_size', type=int, default=int(32))

args = parser.parse_args()


LR = args.lr
N_EPOCH = args.n_epoch
HIDDEN_SIZE = args.hidden_size
BATCH_SIZE = args.batch_size
N_LAYERS = args.n_layers

print_every = 10
plot_every = 10

print(LR, N_EPOCH, HIDDEN_SIZE, N_LAYERS)

timit = TIMIT(args.data, "tr")

model = model_cnn.CNN(N_FEAT, (3, 2), HIDDEN_SIZE, N_LABEL, BATCH_SIZE, N_LAYERS)
if USE_CUDA:
    model = model.cuda()

opt = torch.optim.SGD(model.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss()

def train(inp, target, useful, lens):
    # inp: (BATCH_SIZE x maxlen x N_FEAT)
    # target: (BATCH_SIZE x maxlen)
    hidden = model.init_hidden()
    model.zero_grad()
    output, hidden = model(inp, hidden, lens)

    loss = 0

    for i in range(useful):
        for j in range(lens[i]):
            loss += criterion(output[i][j].view(1, -1), target[i][j])

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

    return loss.data[0] / sum(lens[:useful]), acc / sum(lens[:useful])


def eval_valid():
    loss = 0
    acc = 0
    v_len = len(timit.valid_set)
    for i in range(0, v_len, BATCH_SIZE):
        input, target, useful = timit.get_batch(i, BATCH_SIZE, "va")
        input, target, lens = make_batch(input, target)
        tloss, tacc = batch_eval(input, target, useful, lens)
        loss += tloss * useful
        acc  += tacc * useful

    loss /= v_len
    acc /= v_len

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
        input, target, useful = timit.get_batch(i, BATCH_SIZE)

        input, target, lens = make_batch(input, target)

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

print(all_losses)

torch.save(model.state_dict(), "cnn.pt")
