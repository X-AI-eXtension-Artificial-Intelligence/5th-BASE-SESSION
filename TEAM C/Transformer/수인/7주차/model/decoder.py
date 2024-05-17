import torch
from torch import nn

from blocks.decoder_layer import DecoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 임베딩 레이어 생성
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        # 여러 개의 인코더 레이어를 모아서 관리 
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)]) # n_layer 개수 만큼 반복 생성

        # 선형 레이어 생성 (출력층)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg: 타겟 데이터
        # enc_src: 인코더 출력
        # trg_mas: 타겟 마스크 (패딩 값이나 사용하지 않을 토큰을 처리하기 위한 마스크)
        # src_mask: 소스 마스크 
        trg = self.emb(trg) # 입력 데이터를 임베딩 레이어 통과 

        for layer in self.layers: # 각 디코더 레이어에 순차적으로 통과
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # 최종 디코더 레이어의 출력을 선형 레이어에 통과 (LM head)
        output = self.linear(trg)
        return output # 최종 출력 반환 