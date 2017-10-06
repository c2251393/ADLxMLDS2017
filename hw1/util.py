import time, math
import torch
from torch.autograd import Variable

def make_lab2id(fn, fn2):
    f = open(fn)
    res = {}
    for line in f.readlines():
        lab, id, _ = line.split('\t')
        res[lab] = int(id)
    return res


def make_id2lab(fn, fn2):
    f = open(fn)
    res = {}
    pass

def to_torch_var(xs, ys):
    # (1 x len(xs))
    target = torch.zeros(len(xs)).long()
    for i in range(len(xs)):
        target[i] = ys[i]

    input = torch.Tensor(xs)

    return Variable(input), Variable(target)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def read_data(f, lab_f, lab2id):
    X = {}
    y = {}
    with open(f, 'r') as f:
        for line in f.readlines():
            words = line[:-1].split()
            frame_id = words[0]

            spid, seid, fid = frame_id.split('_')
            id = spid + '_' + seid

            feat = list(map(float, words[1:]))

            if id not in X:
                X[id] = []
            X[id].append(feat)

    if lab_f == None:
        res = list(X.values())
        return res

    with open(lab_f, 'r') as f:
        for line in f.readlines():
            words = line[:-1].split(',')
            frame_id = words[0]

            spid, seid, fid = frame_id.split('_')
            id = spid + '_' + seid

            lab = words[1]
            if id not in y:
                y[id] = []
            y[id].append(lab2id[lab])

    res = []
    for k in X.keys():
        res.append((y[k], X[k]))

    return res

