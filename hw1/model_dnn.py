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
                 batch_size
                 ):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.encoder = nn.Linear(input_size * frame_size, hidden_size)
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        # input: (batch x frame x feat)

        input = input.view(self.batch_size, -1)

        input = F.relu(self.encoder(input))
        input = F.relu(self.W1(input))
        input = F.relu(self.W2(input))
        input = F.relu(self.W3(input))

        output = self.decoder(input)

        return output
