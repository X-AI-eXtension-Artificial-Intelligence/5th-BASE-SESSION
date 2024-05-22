import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    # positonal 정보를 일정한 범위 안의 실수로 제약(정규화)
    def __init__(self, d_embed, max_len=256, device=torch.device("cpu")):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        # forward() 내부에서 slicing해 사용하게 되는데, 
        # 이 encoding이 학습되지 않도록 requires_grad=False을 부여
        encoding.requires_grad = False 
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        # 짝수 index에는 sin함수를, 홀수 index에는 cos함수를 사용
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)


    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out