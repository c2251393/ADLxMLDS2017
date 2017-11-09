import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import *
import random

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
        batch_size = input.size(0)
        self.hc = self.init_hidden(batch_size)
        # input: (batch x len x feat)
        # target_outputs: (batch x MAXLEN)
        # target_lengths: (batch)
        output, self.hc = self.rnn(input, self.hc)
        # output: (batch x len x hidden)
        # hc:     (batch x 1 x hidden)^2
        return output, self.hc

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if USE_CUDA:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch x 1 x hidden)
        # encoder_output: (batch x len x hidden)
        batch_size = hidden.size(0)
        len = encoder_outputs.size(1)
        # hidden: (batch x 1 x hidden)

        hidden = self.attn(hidden)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        # hidden: (batch x 1 x hidden)
        # encoder_output: (batch x hidden x len)
        attn_energies = hidden.bmm(encoder_outputs)
        # (batch x 1 x len)
        attn_energies = attn_energies.squeeze()
        # (batch x len)

        return F.softmax(attn_energies)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 dropout,
                 do_attn=True
                 ):
        super(Decoder, self).__init__()
        self.input_size = 4096
        self.vocab_size = len(lang.word2id)
        self.hidden_size = hidden_size
        self.dropout = dropout

        if do_attn:
            self.attn = Attention(self.hidden_size)
            self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
        else:
            self.attn = None
            self.concat = None
        self.rnn = nn.LSTM(self.hidden_size,
                           self.hidden_size,
                           1,
                           dropout=self.dropout,
                           batch_first=True)
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.W = nn.Linear(self.hidden_size, self.vocab_size)
        self.hc = None

    def forward(self, output, hc, target_outputs, target_lengths, sched_sampling_p=1):
        # target_outputs: (batch x MAXLEN)
        # target_lengths: (batch)
        batch_size = output.size(0)
        max_lens = torch.max(target_lengths).data[0]
        self.hc = hc

        def decode(symbol):
            # symbol: (batch)
            # output: (batch x len x hidden)
            y = self.embed(symbol).unsqueeze(1)
            y, self.hc = self.rnn(y, self.hc)
            # y: (batch x 1 x hidden)

            if self.attn is not None:
                A = self.attn(y, output).unsqueeze(1)
                # A: (batch x 1 x len)
                c = A.bmm(output)
                # c: (batch x 1 x hidden)
                yc = F.tanh(self.concat(torch.cat((y, c), 2)))
                # yc: (batch x 1 x hidden)
            else:
                yc = y

            dec_o = self.W(yc)
            # dec_o: (batch x 1 x vocab_size)
            dec_symbol = dec_o.topk(1, 2)[1]
            # dec_symbol: (batch x 1 x 1)
            return dec_o.squeeze(), dec_symbol.squeeze()

        symbol = Variable(torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]))
        if USE_CUDA:
            symbol = symbol.cuda()
        dec_o = self.W(self.embed(symbol))
        symbol_outs = []
        decoder_outs = []

        # target_outputs: (batch x MAXLEN)
        for i in range(max_lens):
            symbol_outs.append(symbol)
            decoder_outs.append(dec_o)
            if target_outputs is not None and random.random() < sched_sampling_p:
                symbol = target_outputs[:,i]
            dec_o, symbol = decode(symbol)

        decoder_outs = torch.stack(decoder_outs, 1)
        symbol_outs = torch.stack(symbol_outs, 1)

        return decoder_outs, symbol_outs


class S2S(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 dropout=0,
                 do_attn=True
                 ):
        super(S2S, self).__init__()
        self.input_size = 4096
        self.vocab_size = len(lang.word2id)
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.encoder = Encoder(self.hidden_size, self.dropout)
        self.decoder = Decoder(self.hidden_size, self.dropout, do_attn)

    def forward(self, input, target_outputs, target_lengths, sched_sampling_p=1):

        output, hc = self.encoder(input)

        decoder_outs, symbol_outs = self.decoder(output, hc, target_outputs, target_lengths, sched_sampling_p)

        return decoder_outs, symbol_outs
