import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PositionalEncoding

class MemTransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MemTransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError(
                "TransformerEncoder module does not exist in PyTorch 1.1 or lower."
            )
        self.model_type = "MemTransformer"
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.lstm = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(ninp, ntoken)
        self.drop = nn.Dropout(dropout)
        self.nhid = nhid
        self.nlayers = nlayers
        self.fc = nn.Linear(2 * nhid, nhid)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden, has_mask=True):
        if has_mask:
            device = input.device
            if self.src_mask is None or self.src_mask.size(0) != len(input):
                mask = self._generate_square_subsequent_mask(len(input)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        emb = self.drop(self.encoder(input))
        output, hidden = self.lstm(emb, hidden)
        src1 = self.drop(output)
        src2 = self.encoder(input) * math.sqrt(self.ninp)
        src = F.relu(self.fc(torch.cat([src1, src2], dim=2)))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

    def init_hidden(self, bsz):
        weight = next(self.lstm.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )

class BiLSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        tie_weights=False,
        bidirectional=True,
    ):
        super(BiLSTMModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(
            ninp, nhid, nlayers, dropout=dropout, bidirectional=bidirectional
        )
        self.decoder1 = nn.Linear(nhid, ntoken)
        self.decoder2 = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder1.weight = self.encoder.weight
            self.decoder2.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder1.bias)
        nn.init.zeros_(self.decoder2.bias)
        nn.init.uniform_(self.decoder1.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder2.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output1, output2 = torch.chunk(output, 2, dim=2)
        decoded1 = self.decoder1(output1)
        decoded2 = self.decoder2(output2)
        decoded = (decoded1 + decoded2) / 2
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (
            weight.new_zeros(2 * self.nlayers, bsz, self.nhid),
            weight.new_zeros(2 * self.nlayers, bsz, self.nhid),
        )
