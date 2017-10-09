import time, math
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

USE_CUDA = torch.cuda.is_available()

# fn: 48 id ch, fn2: 48 39
def make_lab2id(fn, fn2):
    lab2id = {}
    cur_id = 1
    f2 = open(fn2)
    good_lab = set()
    for line in f2.readlines():
        lab, nlab = line.strip().split('\t')
        if nlab not in lab2id:
            good_lab.add(nlab)
            lab2id[nlab] = cur_id
            cur_id += 1
        lab2id[lab] = lab2id[nlab]

    f = open(fn)
    id2ascii = {}
    for line in f.readlines():
        lab, id, ch = line.strip().split('\t')
        if lab in good_lab:
            id2ascii[lab2id[lab]] = ch
    id2ascii[0] = 'a'
    return lab2id, id2ascii


def pad_feat(seq, max_len, N_FEAT):
    seq += [[0.0 for _ in range(N_FEAT)] for i in range(max_len - len(seq))]
    return seq


def pad_label(seq, max_len):
    seq += [0 for i in range(max_len - len(seq))]
    return seq


def make_batch(xss, yss, N_FEAT):
    # xss: batch_size x len x n_feat
    # yss: batch_size x len
    seq_pairs = sorted(zip(xss, yss), key=lambda p:len(p[0]), reverse=True)
    xss, yss = zip(*seq_pairs)

    lens = [len(xs) for xs in xss]
    max_len = max(lens)
    xss_pad = [pad_feat(xs, max_len, N_FEAT) for xs in xss]
    yss_pad = [pad_label(ys, max_len) for ys in yss]

    # (batch_size x maxlen)
    xss_var = Variable(torch.FloatTensor(xss_pad))
    yss_var = Variable(torch.LongTensor(yss_pad))

    if USE_CUDA:
        xss_var, yss_var = xss_var.cuda(), yss_var.cuda()

    return xss_var, yss_var, lens


def make_batch_te(xss, ids):
    # xss: batch_size x len x n_feat
    # ids: batch_size
    seq_pairs = sorted(zip(xss, ids), key=lambda p:len(p[0]), reverse=True)
    xss, ids = zip(*seq_pairs)

    lens = [len(xs) for xs in xss]
    max_len = max(lens)
    xss_pad = [pad_feat(xs, max_len) for xs in xss]

    # (batch_size x maxlen)
    xss_var = Variable(torch.FloatTensor(xss_pad))

    if USE_CUDA:
        xss_var = xss_var.cuda()

    return xss_var, ids, lens


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

            mean = sum(feat) / len(feat)
            for i in range(len(feat)):
                feat[i] -= mean

            norm = math.sqrt(sum(x*x for x in feat))
            for i in range(len(feat)):
                feat[i] /= norm

            if id not in X:
                X[id] = []
            X[id].append(feat)

    if lab_f == None:
        res = []
        for k in X.keys():
            res.append((X[k], k))
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
        res.append((y[k], X[k], k))

    return res

