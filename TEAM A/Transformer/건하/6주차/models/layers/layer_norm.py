import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)) # 스케일 조정
        self.beta = nn.Parameter(torch.zeros(d_model)) # 편향 및 이동 조정 
        self.eps = eps # 안정적인 수치 계산을 위한 작은 값 사용 # 분산이 0이 되는 것을 방지
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # 마지막 차원에 대한 평균 계산
        var = x.var(-1, unbiased=False, keepdim=True) # 마지막 차원에 대한 분산 계산 

        out = (x-mean) / torch.sqrt(var + self.eps) # 정규화된 출력 계산 
        out = self.gamma * out + self.beta
        return out 