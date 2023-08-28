# refer: github.com/pbcquoc
import numpy as np
import torch, os, math, copy, yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embed = self.embedding(x)
        return embed


class PosisionalEncoder(nn.Module):
    def __init__(self, d_model=768, max_seq_len=256, dropout=0.1):
        super(PosisionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        pe = pe.unsqueeze(0)
        # this makes pe is not trained/updated by optimizer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = self.dropout(x + pe)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=None):
        super(MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        # init mattrix weights for key, query and value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        Parameters:
        -----------
        q: tensor shape `(batch_size, seq_len, d_model)`
        k: tensor shape `(batch_size, seq_len, d_model)`
        v: tensor shape `(batch_size, seq_len, d_model)`
        mask: tensor shape `(batch_size, 1, seq_len)`, the mask of self-attn layer at Decoder
        Return:
        -------
        output: tensor shape `(batch_size, seq_len, d_model)`
        """
        # calculate query, key, value vector from weight mattrix
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # perfrom scale-dot attention op
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, -1e9)
        score = F.softmax(score, -1)
        if self.dropout:
            output = self.dropout(score)
        output = torch.matmul(score, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(output)
        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))

    def forward(self, x):
        out = self.ff(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiheadAttention(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Parameters:
        -----------
        x: tensor shape `(batch_size, seq_len, model_dim)`
        mask: tensor shape `(batch_size, 1, model_dim)` for mask self-attention
        Return:
        -------
        out: tensor shape `(batch_size, seq_len, model_dim)`
        """
        x_norm = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x_norm, x_norm, x_norm, mask))
        x_norm = self.norm_2(x)
        x = x = self.dropout_2(self.ff(x_norm))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.attn_1 = MultiheadAttention(n_heads, d_model, dropout)
        self.attn_2 = MultiheadAttention(n_heads, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Parameters:
        -----------
        x: tensor input of target batch sentences
            shape `(batch_size, seq_len, d_model)`
        encoder_output: tensor output (contextual embedding) of encoder block
            shape `(batch_size, seq_len, d_model)`
        src_mask: tensor mask for encoder output
            shape `(batch_size, 1, seq_len)`
        tgt_mask: tensor for hide the future represented of predicted token from current step
            shape `(batch_size, 1, seq_len)`
        Return:
        -------
        out: tensor, contextual embedding of sentence
            shape `(batch_size, seq_len, d_model)`
        """
        x_norm = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x_norm, x_norm, x_norm, tgt_mask))

        # get corr between current token embedding of decoder with all token embedding from encoder
        x_norm = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x_norm, encoder_output, encoder_output, src_mask))

        x_norm = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x_norm))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layer, dropout=0.1, **kwargs):
        super(Encoder, self).__init__()
        self.N = num_layer
        self.layers = get_clones(EncoderBlock(d_model, n_heads, d_ff, dropout), num_layer)
        self.norm = Norm(d_model)

    def forward(self, x, mask):
        """
        Parameters:
        -----------
        x: tensor, sents representation
            shape `(batch_size, seq_len, d_model)`
        mask: tensor, shape `(batch_size, 1, seq_len)`
        Return:
        -------
        out: tensor, shape `(batch_size, seq_len, d_model)`
        """
        for i in range(self.N):
            x = self.layers[i](x, mask)
        out = self.norm(x)
        return out


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layer, dropout=0.1, **kwargs):
        super(Decoder, self).__init__()
        self.N = num_layer
        self.layers = get_clones(DecoderBlock(d_model, n_heads, d_ff, dropout), num_layer)
        self.norm = Norm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Parameters:
        -----------
        x: tensor, sents representation
            shape `(batch_size, seq_len, d_model)`
        encoder_output: tensor, contextual embedding for input sents
            shape `(batch_size, seq_len, d_model)`
        src_mask: tensor, shape `(batch_size, 1, seq_len)`
            mask for sentence embedding of encoder, the
            source mask is created by checking where the source sequence
            is not equal to a <pad> token. It is 1 where the token is
            not a <pad> token and 0 when it is.
        tgt_mask: tensor, shape `(batch_size, 1, seq_len)`
            mask for token prediction step by step. At step t, the first t
            element is the token index, the remain is set to zero.
        Return:
        -------
        out: contextual embedding of whole predicted sents
        """
        for i in range(self.N):
            x = self.layers[i](x, encoder_output, src_mask, tgt_mask)
        out = self.norm(x)
        return out


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, en_config: dict, de_config: dict, **kwargs):
        super(Transformer, self).__init__()
        assert en_config['d_model'] == de_config['d_model']
        d_model = en_config['d_model']

        self.embed = Embedder(vocab_size, d_model)
        self.encoder_pe = PosisionalEncoder(d_model, en_config['max_seq_len'], en_config['dropout'])
        self.decoder_pe = PosisionalEncoder(d_model, de_config['max_seq_len'], de_config['dropout'])

        self.encoder = Encoder(**en_config)
        self.decoder = Decoder(**de_config)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src_sent, tgt_sent, src_mask, tgt_mask):
        src_sent = self.encoder_pe(self.embed(src_sent))
        tgt_sent = self.decoder_pe(self.embed(tgt_sent))

        encoder_output = self.encoder(src_sent, src_mask)
        decoder_output = self.decoder(tgt_sent, encoder_output, src_mask, tgt_mask)
        out = self.fc(decoder_output)
        return out


if __name__ == '__main__':
    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    en_config = config['en_config']
    de_config = config['de_config']
    vocab_size = 500
    batch_size = 8

    net = Transformer(vocab_size, en_config, de_config)
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    prob_map = net(src_sent=torch.LongTensor(batch_size, en_config['max_seq_len']).random_(0, vocab_size), \
                   tgt_sent=torch.LongTensor(batch_size, de_config['max_seq_len']).random_(0, vocab_size), \
                   src_mask=torch.randn(batch_size, 1, en_config['max_seq_len']), \
                   tgt_mask=torch.randn(batch_size, 1, de_config['max_seq_len']))
    print(prob_map.shape)