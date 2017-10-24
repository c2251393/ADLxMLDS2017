import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *

class CNN(nn.Module):
    def __init__(self,
                 input_size,
                 frame_size,
                 hidden_size, 
                 output_size,
                 batch_size,
                 n_layers=1,
                 dropout=0.0
                 ):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = nn.Linear(input_size * frame_size, hidden_size)
        if USE_CUDA:
            self.Ws = [nn.Linear(hidden_size, hidden_size).cuda() for _ in range(n_layers)]
        else:
            self.Ws = [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        self.decoder = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax()

    def forward(self, input):
        # input: (batch x frame x feat)

        input = F.relu(self.encoder(input.view(self.batch_size, -1)))

        for i in range(self.n_layers):
            input = F.relu(self.Ws[i](input))

        output = self.decoder(input)

        return output
