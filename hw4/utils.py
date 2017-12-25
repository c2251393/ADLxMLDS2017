import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from itertools import count
from collections import namedtuple
import random
import time
import math
import copy
import scipy.misc
import argparse
import os
import skimage
import skimage.io
import skimage.transform
import csv
import _pickle as pickle
from tqdm import tqdm
import h5py
from torchvision.transforms import ToPILImage

pil2img = ToPILImage()

USE_CUDA = torch.cuda.is_available()

EYES = ['<UNK>',
        'gray eyes', 'black eyes', 'orange eyes',
        'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
        'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

HAIR = ['<UNK>',
        'orange hair', 'white hair', 'aqua hair', 'gray hair',
        'green hair', 'red hair', 'purple hair', 'pink hair',
        'blue hair', 'black hair', 'brown hair', 'blonde hair']

EYES_DIM = 50
HAIR_DIM = 50
TAG_DIM = EYES_DIM + HAIR_DIM

VOCAB = set([s.split()[0] for s in EYES[1:]] + [s.split()[0] for s in HAIR[1:]])
glove = {}

for line in open('glove.6B.50d.txt').readlines():
    tokens = line[:-1].split()
    word = tokens[0]
    if word in VOCAB:
        glove[word] = torch.Tensor(list(map(float, tokens[1:])))
print('glove done')

EYES_VEC = [torch.zeros(50).float()] + [glove[s.split()[0]] for s in EYES[1:]]
HAIR_VEC = [torch.zeros(50).float()] + [glove[s.split()[0]] for s in HAIR[1:]]

def to_img(x, fn):
    # x: (3, 64, 64) tensor
    x = x.cpu()
    M = x.max()
    m = x.min()
    x = (x - m) / (M - m)
    x = x*x
    img = pil2img(x)
    img.save(fn)

def get_tag(s):
    # s = '[c] hair [c] eyes'
    w = s.split()
    if w[1] == 'hair':
        w = [w[2], w[3], w[0], w[1]]
    eyes_vec = EYES_VEC[EYES.index("%s %s" % (w[0], w[1]))]
    hair_vec = HAIR_VEC[HAIR.index("%s %s" % (w[2], w[3]))]
    return eyes_vec.unsqueeze(0), hair_vec.unsqueeze(0)

def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def cu(x):
    if USE_CUDA:
        return x.cuda()
    return x
