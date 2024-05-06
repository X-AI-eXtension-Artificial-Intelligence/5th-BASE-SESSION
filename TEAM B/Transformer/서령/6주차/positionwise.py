import torch.nn as nn
import torch.nn.functional as F
from model.ops import init_weight

# Position-wise Feed-forward Network 구현 함수
class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super(PositionWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(params.hidden_dim, params.feed_forward_dim, kernel_size=1) 
        self.conv2 = nn.Conv1d(params.feed_forward_dim, params.hidden_dim, kernel_size=1)
        init_weight(self.conv1)
        init_weight(self.conv2)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, x):
        # 입력 x의 차원 변경: x = [batch size, sentence length, hidden dim] ->  x = [batch size, hidden dim, sentence length]  
        x = x.permute(0, 2, 1)           
        output = self.dropout(F.relu(self.conv1(x)))  
        output = self.conv2(output)                   

        # 출력 차원을 다시 원래 차원으로 변경 : x = [batch size, hidden dim, sentence length] -> x = [batch size, sentence length, hidden dim]
        output = output.permute(0, 2, 1)              
        return self.dropout(output)