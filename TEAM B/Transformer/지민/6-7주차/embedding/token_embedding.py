import math
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed


    def forward(self, x):
         # vocabulary와 d_embed를 사용해 embedding을 생성
        out = self.embedding(x) * math.sqrt(self.d_embed) # scaling
        return out