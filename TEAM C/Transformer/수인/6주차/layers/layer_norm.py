import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # 학습 가능한 스케일링 계수
        self.beta = nn.Parameter(torch.zeros(d_model)) # 학습 가능한 이동 계수
        self.eps = eps # 수치적 안정성을 위한 작은 값

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 마지막 차원에 대한 평균 계산

        var = x.var(-1, unbiased=False, keepdim=True) # 마지막 차원에 대한 분산 계산 (편향))
        # '-1' means last dimension

        # 계산된 평균과 분산을 사용하여 정규화
        out = (x - mean)/torch.sqrt(var + self.eps) 
        # 학습된 매개 변수를 사용하여 정규화된 출력을 스케일링하고 이동 
        out = self.gamma*out + self.beta
        return out