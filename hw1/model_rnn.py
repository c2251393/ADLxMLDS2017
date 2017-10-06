import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size, 
                 output_size,
                 n_layers=1,
                 ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hc):
        output, hc = self.lstm(input.view(1, 1, -1), hc)
        output = self.decoder(output)
        output = self.softmax(output.view(1, -1))
        return output, hc

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        c0 = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return (h0, c0)
