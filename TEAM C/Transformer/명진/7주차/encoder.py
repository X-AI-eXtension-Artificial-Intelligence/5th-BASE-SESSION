import torch
import torch.nn as nn
import math
from model import *
###############################
## encoder & decoder
 # encoder block
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):#incoder input에 적용하는 source mask: padding word를 다른 단어들과 interact못하도록
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #1st skip connection,encoder에서는 xxx같은 들어 값 들어감
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Encoder / 인코더블록*n   
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # n개 반복
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


