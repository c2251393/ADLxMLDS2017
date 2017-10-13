import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *

class DNN(nn.Module):
    def __init__(self,
                 input_size,
                 frame_size,
                 hidden_size, 
                 output_size,
                 batch_size,
                 n_layers=1,
                 dropout=0.0
                 ):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.L = nn.Linear(input_size * frame_size, output_size)

        # self.encoder = nn.Linear(input_size, hidden_size)
        # if USE_CUDA:
            # self.Ws = [nn.Linear(hidden_size, hidden_size).cuda() for _ in range(n_layers)]
        # else:
            # self.Ws = [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        # self.decoder = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax()

    def forward(self, input):
        # input: (batch x frame x feat)

        output = F.relu(self.L(input.view(self.batch_size, -1)))

        # inputs = F.relu(self.encoder(inputs))

        # for i in range(self.n_layers):
            # inputs = F.relu(self.Ws[i](inputs))

        # output = self.decoder(inputs)

        return output
