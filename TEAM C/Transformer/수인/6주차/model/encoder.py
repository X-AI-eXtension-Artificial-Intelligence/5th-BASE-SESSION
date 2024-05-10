from torch import nn

from models.blocks.encoder_layer import EncoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 임베딩 레이어 생성
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        # 여러 개의 인코더 레이어를 모아서 관리 
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)]) # n_layers 개수만큼 반복 생성

    def forward(self, x, src_mask):
        # src_mask: 패딩 값이나 사용하지 않을 토큰을 처리하기 위한 마스크
        x = self.emb(x) # 입력 데이터를 임베딩 레이어 통과

        for layer in self.layers: # 각 인코더 레이어에 순차적으로 통과
            x = layer(x, src_mask)

        return x # 최종 인코더 레이어의 출력 반환 