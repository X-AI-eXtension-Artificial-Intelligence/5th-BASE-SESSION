import torch
from torch import nn

from models.model.decoder import Decoder
from models.model.encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()

        # 패딩 인덱스 및 시작 토큰 인덱스
        self.src_pad_idx = src_pad_idx # 소스 채딩 인덱스
        self.trg_pad_idx = trg_pad_idx # 타겟 패딩 인덱스
        self.trg_sos_idx = trg_sos_idx # 타겟 시작 토큰 인덱스
        self.device = device 
        # 인코더 및 디코더 생성
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

        self.decoder = Decoder(d_model=d_model,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               dec_voc_size=dec_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) # 소스 마스크 생성
        trg_mask = self.make_trg_mask(trg) # 타겟 마스크 생성
        enc_src = self.encoder(src, src_mask) # 인코더 통과
        output = self.decoder(trg, enc_src, trg_mask, src_mask) # 디코더 통과
        return output

    # 소스 문장의  토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 0 0
        1 1 1 0 0
        """
        trg_len = trg.shape[1] # 타겟 시퀀스 길이
        # 아래 삼각 행렬 생성 (미래 단어를 무시하도록 함)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """
        trg_mask = trg_pad_mask & trg_sub_mask # 패딩 마스크와 삼각 마스크를 결함
        return trg_mask