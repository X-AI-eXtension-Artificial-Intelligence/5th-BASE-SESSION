from torch import nn 

from models.blocks.encoder_layer import EncoderLayer 
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len = max_len.
                                        vocab_size = enc_voc_size,
                                        drop_prob = drop_prob,
                                        device = device)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                  n_head = n_head,
                                                  drop_prob = drop_prob)
                                                for _ in range(n_layers)]) # Encoder Layer를 n_layers만큼 생성함
    
    def forward(self, x, src_mask):
        x = self.emb(x) # 입력 데이터에 대한 임베딩 적용 

        for layer in self.layers:
            x = layer(x, src_mask) # 각 Encoder Layer를 순차적으로 적용함

        return x 