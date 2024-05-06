import torch
from torch import nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        '''
        param d_model: 모델의 차원
        param max_len: 최대 시퀀스 길이
        param device: 하드웨어 장치 설정
        '''
        super(PositionalEncoding, self).__init__()

        # 입력 행렬과 같은 크기 (입력 행렬에 더하기 위해서)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 기울기 계산 필요 X

        pos = torch.arange(0, max_len, device=device) 
        pos = pos.float().unsqueeze(dim=1) 
        # 1d -> 2차원으로 차원 확정하여 broadcasting 연산 수월할 수 있도록
            '''
        [max_len(3), 1] 
            [[0]
            [1.]
            [2]]
            '''

        _2i = torch.arange(0, d_model, step=2, device=device).float() # i는 d_model의 index 의미 # step2는 2*i와 같음

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encodingp[:, 1::2] = torch.cos(pos / 10000 ** ((_2i / d_model)))
    
    def forward(self, x):
        # max_len = 512 # d_model = 512
        batch_size, seq_len = x.size() # batch_size = 128, seq_len = 30

        return self.encoding[:seq_len, :] # tok_emb와 더해져 [128, 30, 512]