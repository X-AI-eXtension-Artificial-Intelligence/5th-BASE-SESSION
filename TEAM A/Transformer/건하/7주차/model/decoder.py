import torch
from torch import nn 

from models.blocks.encoder_layer import DecoderLayer 
from models.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len
                                        vocab_size=dec_voc_size,
                                        device=device)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                        for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        '''
        trg_mask -> 타켓 마스킹 (Decoder Attention)
        src_mask -> 소스 마스킹 (Encoder-Deocder Attention)
        '''
        trg = self.emb(trg) # 타켓 시퀀스에 대한 임베딩 적용 

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.linear(trg)
        
        return output 