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
            self.concat = nn.Linear(2*self.hidden_size, self.hidden_size)
        else:
            self.attn = None
            self.concat = None
        self.rnn = nn.LSTM(self.hidden_size + self.embed_size,
                           self.hidden_size,
                           self.n_layers,
                           dropout=self.dropout,
                           batch_first=True)
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.W = nn.Linear(self.hidden_size*2, self.vocab_size)
        self.hc = None

    def forward(self,
                output,
                hc,
                target_outputs,
                target_lengths,
                sched_sampling_p=1,
                beam_search=-1
                ):
        # target_outputs: (batch x MAXLEN)
        # target_lengths: (batch)
        batch_size = output.size(0)
        max_lens = MAXLEN
        # max_lens = torch.max(target_lengths).data[0]
        self.hc = hc

        def decode(symbol, hidden, prv_context):
            # symbol: (batch)
            # output: (batch, len, hidden)
            embed = self.embed(symbol).view(-1, 1, self.embed_size)
            # embed: (batch x 1 x embed)
            y = torch.cat((embed, prv_context), 2)
            # y: (batch x 1 x (embed + hidden))
            y, hidden = self.rnn(y, hidden)
            # y: (batch x 1 x hidden)

            A = self.attn(y, output).unsqueeze(1)
            # A: (batch x 1 x len)
            c = A.bmm(output)
            # c: (batch x 1 x hidden)
            yc = torch.cat((y, c), 2)
            # yc: (batch x 1 x hidden*2)

            dec_o = self.W(yc)
            # dec_o: (batch x 1 x vocab_size)
            dec_symbol = dec_o.topk(1, 2)[1]
            # dec_symbol: (batch x 1 x 1)
            return dec_o.squeeze(), dec_symbol.squeeze(), hidden, c

        symbol = Variable(torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]))
        context = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        dec_o = Variable(torch.zeros(batch_size, self.vocab_size))
        dec_o[:, SOS_TOKEN] = 1.0
        if USE_CUDA:
            symbol = symbol.cuda()
            context = context.cuda()
            dec_o = dec_o.cuda()
        # symbol: (batch)
        # context: (batch, 1, hidden)
        # dec_o: (batch, vocab)

        if beam_search == -1:
            symbol_outs = []
            decoder_outs = []

            # target_outputs: (batch x MAXLEN)
            for i in range(max_lens):
                symbol_outs.append(symbol)
                decoder_outs.append(dec_o)
                if target_outputs is not None and random.random() < sched_sampling_p:
                    symbol = target_outputs[:,i]
                dec_o, symbol, self.hc, context = decode(symbol, self.hc, context)

            decoder_outs = torch.stack(decoder_outs, 1)
            # (batch x max_lens x vocab)
            symbol_outs = torch.stack(symbol_outs, 1)
            # (batch x max_lens)

            return decoder_outs, symbol_outs
        else:
            # Beam search: batch=1

            candidates = []
            tmp_candidates = [(1.0, [dec_o], [symbol], self.hc)]
            print("beam search ", beam_search)
            print(dec_o.size())
            for i in range(max_lens):
                tmp_candidates.sort(key=lambda p: p[0], reverse=True)
                candidates = tmp_candidates[:beam_search]
                tmp_candidates = []
                for (prob, decoder_outs, symbol_outs, hidden) in candidates:
                    last_dec_o = decoder_outs[-1]
                    last_symbol = symbol_outs[-1]
                    dec_o, _, hidden = decode(last_symbol, hidden)
                    # dec_o: (vocab_size)

                    topk_symbols = dec_o.topk(beam_search)[1]
                    p_sigma = F.log_softmax(dec_o)
                    # (beam_search)
                    for j in range(beam_search):
                        next_symbol = topk_symbols[j]
                        next_prob = prob + p_sigma[next_symbol.data[0]].data[0]
                        tmp_candidates.append((
                            next_prob,
                            decoder_outs + [dec_o],
                            symbol_outs + [next_symbol],
                            hidden))

            (_, decoder_outs, symbol_outs, _) = candidates[0]
            decoder_outs = torch.stack(decoder_outs).view(1, -1)
            symbol_outs = torch.stack(symbol_outs).view(1, -1)

            return decoder_outs, symbol_outs


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

        output, hc = self.encoder(input)

        decoder_outs, symbol_outs = self.decoder(output, hc, target_outputs, target_lengths, sched_sampling_p, beam_search)

        return decoder_outs, symbol_outs
