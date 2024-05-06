import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, source, target):
        # Encoder에 source sequence를 넣어 Encoder의 출력을 얻음
        encoder_output = self.encoder(source)     
        # Decoder에 target sequence, source sequence, Encoder의 출력을 넣어 최종 출력과 attention map을 얻음                   
        output, attn_map = self.decoder(target, source, encoder_output)  
        return output, attn_map

    def count_params(self):
        # 모델의 학습 가능한 파라미터 수를 계산함
        return sum(p.numel() for p in self.parameters() if p.requires_grad)