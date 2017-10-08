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
parser.add_argument('model', default='rnn.pt',
                    help='model file')
args = parser.parse_args()

timit = TIMIT(args.data, "te")

