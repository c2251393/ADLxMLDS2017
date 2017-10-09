import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *

class BRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size, 
                 output_size,
                 batch_size,
                 n_layers=1,
                 dropout=0.0
                 ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size,
                            hidden_size // 2,
                            n_layers,
                            batch_first=True,
                            dropout=self.dropout,
                            bidirectional=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hc, lens):
        # input: (batch x maxlen x feat)
        input_p = pack_padded_sequence(input, lens, batch_first=True)
        output_p, hc = self.lstm(input_p, hc)
        output, _ = pad_packed_sequence(output_p, batch_first=True)

        output.contiguous()

        output = self.decoder(output.view(-1, self.hidden_size))
        output = self.softmax(output)

        output = output.view(self.batch_size, -1, self.output_size)

        return output, hc

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size // 2))
        c0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size // 2))
        if USE_CUDA:
            h0, c0 = h0.cuda(), c0.cuda()
        return (h0, c0)
