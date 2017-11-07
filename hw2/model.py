import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *

class Encoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout
                 ):
        super(Encoder, self).__init__()
        self.input_size = 4096
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           1,
                           dropout=self.dropout,
                           batch_first=True)
        self.hc = None

    def forward(self, input):
        batch_size = input.size()[0]
        self.hc = self.init_hidden(batch_size)
        # input: (batch x len x feat)
        # target_outputs: (batch x n_cap x MAXLEN)
        # target_lengths: (batch x n_cap)
        output, self.hc = self.rnn(input, self.hc)
        # output: (batch x len x feat)
        # hc:     (batch x 1 x feat)^2
        return output, self.hc

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout
                 ):
        super(Decoder, self).__init__()
        self.input_size = 4096
        self.vocab_size = len(lang.word2id)
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.rnn = nn.LSTM(self.hidden_size,
                           self.hidden_size,
                           1,
                           dropout=self.dropout,
                           batch_first=True)
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.vocab_size)
        self.hc = None

    def forward(self, output, hc, target_outputs, target_lengths):
        batch_size = output.size()[0]
        max_lens = torch.max(target_lengths).data[0]
        self.hc = hc

        symbol_outs = []
        decoder_outs = []

        def decode(symbol):
            # symbol: (batch)
            x = self.embed(symbol).unsqueeze(1)
            # x: (batch x 1 x hidden)
            x, self.hc = self.rnn(x, self.hc)
            dec_o = self.W(x)
            # dec_o: (batch x 1 x vocab_size)
            dec_symbol = dec_o.topk(1, 2)[1]
            # dec_symbol: (batch x 1 x 1)
            return dec_o.squeeze(), dec_symbol.squeeze()

        symbol = Variable(torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]))
        if USE_CUDA:
            symbol = symbol.cuda()
        dec_o = self.W(self.embed(symbol))

        for i in range(max_lens):
            symbol_outs.append(symbol)
            decoder_outs.append(dec_o)
            dec_o, symbol = decode(symbol)

        decoder_outs = torch.stack(decoder_outs, 1)
        symbol_outs = torch.stack(symbol_outs, 1)

        return decoder_outs, symbol_outs


class S2S(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 dropout=0
                 ):
        super(S2S, self).__init__()
        self.input_size = 4096
        self.vocab_size = len(lang.word2id)
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.encoder = Encoder(self.hidden_size, self.dropout)
        self.decoder = Decoder(self.hidden_size, self.dropout)

    def forward(self, input, target_outputs, target_lengths):

        output, hc = self.encoder(input)

        decoder_outs, symbol_outs = self.decoder(output, hc, target_outputs, target_lengths)

        return decoder_outs, symbol_outs
