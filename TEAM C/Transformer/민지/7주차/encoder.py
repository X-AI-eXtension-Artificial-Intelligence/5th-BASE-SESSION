#EncoderBlock
import torch
import torch.nn as nn 
from MultiheadAttention import MultiHeadAttentionBlock, FeedForwardBlock, ResidualConnection, LayerNormalization


class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_atttention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        #attention 부분 skip connection 실행
        x = self.residual_connections[0](x, lambda x: self.self_atttention_block(x,x,x,src_mask))
        #feed forward 부분 skip connection 실행
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

#Encoder
class Encoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList) -> None : 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)