import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.ops import init_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        assert params.hidden_dim % params.n_head == 0
        self.attentions = nn.Modulist([SelfAttention(params) for _ in range(params.n_head)])
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, query, key, value, mask=None):
        self_attention = [attention(query, key, value) for attention in self.atentions]
        weighted_vs = [weighted_v[0] for weighted_v in self_attention]
        attentions = [weighted_v[1] for weighted_v in self_attention]

        weighted_v = torch.cat(weighted_vs, dim=-1)
        output = self.dropout(self.o_w(weighted_v))

        return output, attentions
    

class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attnention_dim = params.hidden_dim // params.n_head

        self.q_w = nn.linear(self.hidden_dim, self.attnention_dim, bias=False)
        self.k_w = nn.linear(self.hidden_dim, self.attnention_dim, bias=False)
        self.v_w = nn.linear(self.hidden_dim, self.attnention_dim, bias=False)
        init_weight(self.q_w)
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attnention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)

        self_attention = torch.bmm(q,k.permute(0,2,1))
        self_attention = self_attention / self.scale_factor

        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        attention_score = F.softmax(self_attention, dim=-1)
        norm_attention_score = self.dropout(attention_score)

        weighted_v = torch.bmm(norm_attention_score, v)

        return self.dropout(weighted_v), attention_score