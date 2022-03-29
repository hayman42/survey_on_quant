from ast import Mult
import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.attention_head_size = hidden_size // n_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_q, x_k, x_v, att_mask):
        q, k, v = self.q(x_q), self.k(x_k), self.v(x_v)
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.attention_head_size).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.attention_head_size).permute(0, 2, 3, 1)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.attention_head_size).permute(0, 2, 1, 3)
        att_scores = torch.matmul(q, k) / math.sqrt(self.attention_head_size) + att_mask
        att_probs = F.softmax(att_scores, dim=-1)
        context = torch.matmul(att_probs, v).permute(0, 2, 1, 3).contiguous()
        return context.reshape(context.shape[0], context.shape[1], self.hidden_size)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u)
        s = s * s
        s = s.mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s)
        return self.gamma * x + self.beta


class FeedForward(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return F.relu(self.linear(x))


class Encoder(nn.Module):
    def __init__(self, hidden_size, n_heads):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(hidden_size, n_heads)
        self.LayerNorm = LayerNorm(hidden_size)
        self.FeedForward = FeedForward(hidden_size, hidden_size)

    def forward(self, x):
        att = self.Attention(x, x, x, 0)
        x = x + self.LayerNorm(att)
        return x + self.FeedForward(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_heads):
        super(Encoder, self).__init__()
        self.MaskedAttention = MultiHeadAttention(hidden_size, n_heads)
        self.Attention = MultiHeadAttention(hidden_size, n_heads)
        self.LayerNorm = LayerNorm(hidden_size)
        self.FeedForward = FeedForward(hidden_size, output_size)

    def forward(self, x, encoder_output, att_mask):
        att = self.MaskedAttention(x, x, x, att_mask)
        x = x + self.LayerNorm(att)
        att = self.Attention(encoder_output, encoder_output, x)
        x = x + self.LayerNorm(att)
        return x + self.FeedForward(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 1) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, :] = position * div_term
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, n_tokens, d_model, n_head, n_layers, device):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.pe = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([Encoder(d_model, n_head) for _ in range(n_layers)])
        self.decoder = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        x = self.pe(self.embedding(x))
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
        return self.decoder(x)
