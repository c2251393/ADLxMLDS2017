import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *

class RES(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, n_layers):
        super(RES, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

        convs = []
        for i in range(n_layers):
            convs.append(nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
            convs.append(nn.BatchNorm2d(out_ch))
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(*convs)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, input):
        resi = input
        input = self.conv1(input)
        input = self.bn1(input)
        input = self.relu(input)
        input = self.convs(input)

        if self.proj is not None:
            resi = self.proj(resi)

        output = resi + input
        return output


class RESR(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 batch_size,
                 n_layers,
                 dropout
                 ):
        super(RESR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.dropout = dropout

        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          n_layers,
                          batch_first=True,
                          dropout=self.dropout)

        self.res1 = RES(1, 16, 3, 6)
        self.res2 = RES(16, 8, 3, 2)
        self.res3 = RES(8, 4, 3, 2)
        self.res4 = RES(4, 2, 3, 2)

        self.W = nn.Linear(2 * self.hidden_size, self.output_size)

    def forward(self, input, hc, lens):
        # input: (batch x maxlen x feat)
        input_p = pack_padded_sequence(input, lens, batch_first=True)
        output_p, hc = self.rnn(input_p, hc)
        output, _ = pad_packed_sequence(output_p, batch_first=True)
        # output: (batch x maxlen x 128)

        output = output.unsqueeze(1)
        # output: (batch x 1 x maxlen x 128)
        output = self.res1(output)
        # output: (batch x 16 x maxlen x 128)
        output = self.res2(output)
        # output: (batch x 8 x maxlen x 128)
        output = self.res3(output)
        # output: (batch x 4 x maxlen x 128)
        output = self.res4(output)
        # output: (batch x 2 x maxlen x 128)

        output = torch.cat((output[:,0,:,:], output[:,1,:,:]), dim=2)

        output = self.W(output)

        return output, hc

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size))
        if USE_CUDA:
            h0, c0 = h0.cuda(), c0.cuda()
        # return h0
        return (h0, c0)
