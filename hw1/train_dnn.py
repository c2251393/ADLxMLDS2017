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
parser.add_argument('--lr', type=float, default=float(0.1))
parser.add_argument('--n_epoch', type=int, default=int(3))
parser.add_argument('--hidden_size', type=int, default=int(20))
parser.add_argument('--n_layers', type=int, default=int(1))
parser.add_argument('--batch_size', type=int, default=int(32))
parser.add_argument('--frame_size', type=int, default=int(17))
parser.add_argument('--window_size_x', type=int, default=int(3))
parser.add_argument('--window_size_y', type=int, default=int(2))
parser.add_argument('--dropout', type=float, default=int(0.0))

args = parser.parse_args()

LR = args.lr
N_EPOCH = args.n_epoch
HIDDEN_SIZE = args.hidden_size
FRAME_SIZE = args.frame_size
WINDOW_SIZE = (args.window_size_x, args.window_size_y)
BATCH_SIZE = args.batch_size
N_LAYERS = args.n_layers
DROPOUT = args.dropout

print_every = 100
plot_every = 10

print(args.feat, LR, N_EPOCH, HIDDEN_SIZE, N_LAYERS, BATCH_SIZE, FRAME_SIZE, WINDOW_SIZE, DROPOUT)

timit = TIMIT(args.data, "tr", args.feat)

model = model_dnn.DNN(timit.N_FEAT, FRAME_SIZE, HIDDEN_SIZE, timit.N_LABEL, BATCH_SIZE, N_LAYERS, DROPOUT)

if USE_CUDA:
    model = model.cuda()

opt = torch.optim.Adam(model.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss(timit.label_wt())
# criterion = nn.CrossEntropyLoss()

def train(inp, target, useful):
    # inp: (BATCH_SIZE x frame x N_FEAT)
    # target: (BATCH_SIZE)
    if USE_CUDA:
        inp.cuda()
        target.cuda()
    model.train()
    opt.zero_grad()
    output = model(inp)

    loss = criterion(output[:useful], target[:useful])

    loss.backward()
    opt.step()

    return loss.data[0]


def batch_eval(inp, target, useful):
    # inp: (BATCH_SIZE x maxlen x N_FEAT)
    # target: (BATCH_SIZE x maxlen)
    if USE_CUDA:
        inp.cuda()
        target.cuda()
    output = model(inp)

    loss = criterion(output[:useful], target[:useful]).data[0]

    my_y = output.max(1)[1]
    acc = sum(my_y[:useful] == target[:useful]).data[0] / useful

    return loss, acc


def eval_valid():
    loss = 0
    acc = 0
    v_len = len(timit.valid_set)
    tot_len = 0
    model.eval()
    for i in range(0, v_len, BATCH_SIZE):
        xss, yss, useful = timit.get_flat_batch(i, BATCH_SIZE, FRAME_SIZE, "va")
        X = Variable(torch.Tensor(xss))
        Y = Variable(torch.LongTensor(yss))
        tloss, tacc = batch_eval(X, Y, useful)
        loss += tloss * useful
        acc  += tacc * useful
        tot_len += useful

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
    random.shuffle(timit.tr_idx_set)
    for i in range(0, len(timit.tr_idx_set), BATCH_SIZE):
        xss, yss, useful = timit.get_flat_batch(i, BATCH_SIZE, FRAME_SIZE)

        X = Variable(torch.Tensor(xss))
        Y = Variable(torch.LongTensor(yss))

        loss = train(X, Y, useful)

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
    torch.save(
        model.state_dict(),
        os.path.join("models", 
        ("dnn.e%d.h%d.b%d.l%d.f%d.pt" % (epoch, HIDDEN_SIZE, BATCH_SIZE, N_LAYERS, FRAME_SIZE))))

print(all_losses)

