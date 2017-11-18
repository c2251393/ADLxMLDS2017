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
                 n_layers,
                 dropout
                 ):
        super(Encoder, self).__init__()
        self.input_size = 4096
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           self.n_layers,
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
        # hc:     (batch x layer x hidden)^2
        return output, self.hc

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
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

        hidden = self.attn(hidden)
        # hidden: (batch x 1 x hidden)
        encoder_outputs = encoder_outputs.transpose(1, 2)
        # encoder_output: (batch x hidden x len)
        attn_energies = hidden.bmm(encoder_outputs)
        # (batch x 1 x len)
        attn_energies = attn_energies.squeeze()
        # (batch x len)

        return F.softmax(attn_energies)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 embed_size,
                 n_layers,
                 dropout,
                 do_attn=True
                 ):
        super(Decoder, self).__init__()
        self.input_size = 4096
        self.vocab_size = len(lang.word2id)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        if do_attn:
            self.attn = Attention(self.hidden_size)
            self.rnn = nn.LSTM(self.hidden_size + self.embed_size,
                               self.hidden_size,
                               self.n_layers,
                               dropout=self.dropout,
                               batch_first=True)
            self.W = nn.Linear(self.hidden_size * 2, self.vocab_size)
        else:
            self.attn = None
            self.rnn = nn.LSTM(self.embed_size,
                               self.hidden_size,
                               self.n_layers,
                               dropout=self.dropout,
                               batch_first=True)
            self.W = nn.Linear(self.hidden_size, self.vocab_size)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.hc = None

    def forward(self, symbol, hidden, prv_context, encoder_outputs):
        # symbol: (batch)
        # encoder_outputs: (batch, len, hidden)
        embed = self.embed(symbol).view(-1, 1, self.embed_size)
        # embed: (batch x 1 x embed)

        if self.attn:
            y = torch.cat((embed, prv_context), 2)
            # y: (batch x 1 x (embed + hidden))
            y, hidden = self.rnn(y, hidden)
            # y: (batch x 1 x hidden)

            A = self.attn(y, encoder_outputs).unsqueeze(1)
            # A: (batch x 1 x len)
            c = A.bmm(encoder_outputs)
            # c: (batch x 1 x hidden)
            yc = torch.cat((y, c), 2)
            # yc: (batch x 1 x hidden*2)

            dec_o = self.W(yc)
            # dec_o: (batch x 1 x vocab_size)
        else:
            y = embed
            y, hidden = self.rnn(y, hidden)
            # y: (batch x 1 x hidden)
            c = prv_context
            dec_o = self.W(y)
            # dec_o: (batch x 1 x vocab_size)

        dec_symbol = dec_o.topk(1, 2)[1]
        # dec_symbol: (batch x 1 x 1)
        return dec_o.squeeze(), dec_symbol.squeeze(), hidden, c


class S2S(nn.Module):
    def __init__(self,
                 hidden_size=256,
                 embed_size=256,
                 n_layers=1,
                 dropout=0,
                 do_attn=True
                 ):
        super(S2S, self).__init__()
        self.input_size = 4096
        self.vocab_size = VOCAB
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = Encoder(self.hidden_size, self.n_layers, self.dropout)
        self.decoder = Decoder(self.hidden_size, self.embed_size, self.n_layers, self.dropout, do_attn)

    def forward(self, input, target_outputs, target_lengths,
                sched_sampling_p=1,
                beam_search=-1):

        encoder_outputs, hc = self.encoder(input)

        decoder_outs, symbol_outs = self.decoder(encoder_outputs, hc, target_outputs, target_lengths, sched_sampling_p, beam_search)

        return decoder_outs, symbol_outs
