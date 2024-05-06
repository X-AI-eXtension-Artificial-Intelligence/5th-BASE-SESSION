import torch 
from torch import nn

from models.model.decoder import Decoder 
from models.model.encoder import Encoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
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
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

    def make_src_mask(self, src): # 입력 데이터 중 패딩된 부분을 식별하고, 실제 데이터 부분에만 집중할 수 있도록함 
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2) # src Tensor [batch_size, sequence_length]
        # unsqueeze(1) -> [batch_size, 1, sequence_length] 🎈 첫번째 1은 Attention Head에 대한 차원을 의미
        ## unsqueeze(2) ->  [batch_size, 1, 1, sequence_length] 🎈 두번째 1은 Query의 시퀀스 길이 🎈 이것이 최종 형태임 
        return src_mask # 패딩 인덱스와 같지 않으면 True(실제 유효한 데이터 위치) <-> 패딩 인덱스와 같다면 False(패딩된 위치)

    def make_trg_mask(self, trg): # 디코더에서 사용되는 타켓 시권스에 대한 마스킹을 생성하는 과정 
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3) # [batch_size, 1, sequence_length, 1] # 패딩 마스크 # 패딩 토큰이 아닌 부분에는 True, 패딩 토큰에는 False가 설정
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)  # 순차 마스크 
        # triangular lower 마스킹을 생성하여 디코더가 현재 위치보다 미래에 있는 토큰을 보지 못하게 함
        # 위에서 trg_len x trg_len인 하삼각 행렬을 생성함 / 여기서 대각선과 그 아래는 1로 채워짐 그 위는 0으로 채워짐
        trg_mask = trg_pad_mask & trg_sub_mask # 패딩 마스크와 순차 마스크가 모두 True이면 최종적으로 True 되도록 함 
        return trg_mask 