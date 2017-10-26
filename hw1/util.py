import time, math
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
import numpy as np
from scipy import stats

USE_CUDA = torch.cuda.is_available()

# fn: 48 id ch, fn2: 48 39
def make_lab2id(fn, fn2):
    lab2id = {}
    lad2nlab = {}
    f2 = open(fn2)
    good_lab = set()
    cur_id = 1
    for line in f2.readlines():
        lab, nlab = line.strip().split('\t')
        if nlab not in lab2id:
            good_lab.add(nlab)
            lab2id[nlab] = cur_id
            cur_id += 1
        lab2id[lab] = lab2id[nlab]

    id2ascii = {}
    f = open(fn)
    for line in f.readlines():
        lab, id, ch = line.strip().split('\t')
        if lab in good_lab:
            id2ascii[lab2id[lab]] = ch
    id2ascii[0] = 'a'
    return lab2id, id2ascii


def get_pad(seq, i, lsz, rsz, pad):
    l = i - lsz
    r = i + rsz + 1
    LP, RP = [], []
    if l < 0:
        LP = [pad for _ in range(-l)]
        l = 0
    if r > len(seq):
        RP = [pad for _ in range(r - len(seq))]
        r = len(seq)
    res = copy.deepcopy(seq[l:r])
    return LP + res + RP


def pad_feat(seq, max_len, N_FEAT):
    seq += [[0.0 for _ in range(N_FEAT)] for i in range(max_len - len(seq))]
    return seq


def pad_label(seq, max_len):
    seq += [0 for i in range(max_len - len(seq))]
    return seq


def make_batch(xss, yss, ids, N_FEAT):
    # xss: batch_size x len x n_feat
    # yss: batch_size x len
    seq_pairs = sorted(zip(xss, yss, ids), key=lambda p:len(p[1]), reverse=True)
    xss, yss, ids = zip(*seq_pairs)

    lens = [len(xs) for xs in xss]
    max_len = max(lens)
    xss_pad = [np.pad(xs, ((0, max_len - len(xs)), (0, 0)), 'constant', constant_values=0) for xs in xss]
    xss_pad = np.array(xss_pad)
    yss_pad = [np.pad(ys, (0, max_len - len(ys)), 'constant', constant_values=0) for ys in yss]
    yss_pad = np.array(yss_pad)

    # (batch_size x maxlen)
    xss_var = Variable(torch.from_numpy(xss_pad))
    xss_var = xss_var.type(torch.FloatTensor)
    yss_var = Variable(torch.from_numpy(yss_pad))
    yss_var = yss_var.type(torch.LongTensor)

    if USE_CUDA:
        xss_var, yss_var = xss_var.cuda(), yss_var.cuda()

    return xss_var, yss_var, ids, lens


def make_batch_te(xss, ids, N_FEAT):
    # xss: batch_size x len x n_feat
    # ids: batch_size
    seq_pairs = sorted(zip(xss, ids), key=lambda p:len(p[0]), reverse=True)
    xss, ids = zip(*seq_pairs)

    lens = [len(xs) for xs in xss]
    max_len = max(lens)
    xss_pad = [np.pad(xs, ((0, max_len - len(xs)), (0, 0)), 'constant', constant_values=0) for xs in xss]
    xss_pad = np.array(xss_pad)

    # (batch_size x maxlen)
    xss_var = Variable(torch.from_numpy(xss_pad))
    xss_var = xss_var.type(torch.FloatTensor)

    if USE_CUDA:
        xss_var = xss_var.cuda()

    return xss_var, ids, lens


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def read_data(fs, lab_f, lab2id):
    print(fs, lab_f)
    X = {}
    y = {}

    frames = {}
    for f in fs:
        with open(f, 'r') as fp:
            for line in fp.readlines():
                words = line[:-1].split()
                frame_id = words[0]
                if frame_id not in frames:
                    frames[frame_id] = []
                frames[frame_id].extend(map(float, words[1:]))
    print("tot # of frames: ", len(frames))

    frame_ids = sorted(list(frames.keys()), key=lambda s: int(s.split('_')[2]))
    for frame_id in frame_ids:
        feat = frames[frame_id]
        spid, seid, fid = frame_id.split('_')
        id = spid + '_' + seid
        feat = stats.zscore(feat)
        if id not in X:
            X[id] = []
        X[id].append(feat)

    if lab_f == None:
        res = []
        for k in X.keys():
            res.append((np.array(X[k]), k))
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
        res.append((np.array(y[k]), np.array(X[k]), k))

    return res

