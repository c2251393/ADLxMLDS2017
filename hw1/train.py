import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import glob
from util import *
import random
import model_rnn

parser = argparse.ArgumentParser(description='')
parser.add_argument('data', default='./data/',
                    help='data folder')
parser.add_argument('--lr', type=float, default=float(0.1))
parser.add_argument('--n_epoch', type=int, default=int(10000))
parser.add_argument('--hidden_size', type=int, default=int(100))
args = parser.parse_args()


def random_batch(batch_size, pairs):
    input_seqs = []
    target_seqs = []
    for i in range(batch_size):
        (ys, xs) = random.choice(pairs)
        input_seqs.append(xs)
        target_seqs.append(ys)
    max_len = max(map(len, input_seqs))
    for i in range(batch_size):
        pass

LR = args.lr
N_EPOCH = args.n_epoch
HIDDEN_SIZE = args.hidden_size
N_FEAT = 39
N_LABEL = 49
print_every = 100
valid_every = 2000
plot_every = 10

print(LR, N_EPOCH, HIDDEN_SIZE)

train_f = args.data + "mfcc/train.ark"
train_lab_f = args.data + "label/train.lab"
lab2id = make_lab2id(args.data + "48phone_char.map", args.data + "/phones/48_39.map")

# [(targets, inputs)]
pairs = read_data(train_f, train_lab_f, lab2id)
random.shuffle(pairs)
tr_set, valid_set = pairs[100:], pairs[:100]

def random_train_inst():
    ys, xs = random.choice(tr_set)
    return to_torch_var(xs, ys)


rnn = model_rnn.RNN(N_FEAT, HIDDEN_SIZE, N_LABEL, 1)
opt = torch.optim.SGD(rnn.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss()

def train(inp, target):
    # print(inp)
    # print(target)
    hidden = rnn.init_hidden()
    rnn.zero_grad()
    loss = 0

    for c in range(len(inp)):
        output, hidden = rnn(inp[c], hidden)
        loss += criterion(output, target[c])

    loss.backward()
    opt.step()

    return loss.data[0] / len(inp)

def evaluate(inp, target):
    hidden = rnn.init_hidden()
    loss = 0
    for c in range(len(inp)):
        output, hidden = rnn(inp[c], hidden)
        loss += criterion(output, target[c])
    return loss.data[0] / len(inp)

def eval_valid():
    loss = 0
    for i in range(len(valid_set)):
        ys, xs = valid_set[i]
        loss += evaluate(*to_torch_var(xs, ys))
    return loss / len(valid_set)


start = time.time()
loss_avg = 0
all_losses = []
loss_tot = 0

for epoch in range(1, N_EPOCH + 1):
    input, target = random_train_inst()
    loss = train(input, target)
    loss_avg += loss
    loss_tot += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f %.4f]' % (time_since(start), epoch, epoch / N_EPOCH * 100, loss, loss_tot / epoch))

    if epoch % valid_every == 0:
        print('  VALID LOSS %f' % (eval_valid()))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

print(all_losses)

torch.save(rnn.state_dict(), "rnn.pt")
