from torch import nn

### 3.3 Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 첫 번째 선형 레이어
        self.linear2 = nn.Linear(hidden, d_model) # 두 번째 선형 레이어
        self.relu = nn.ReLU() # ReLU 활성화 함수
        self.dropout = nn.Dropout(p=drop_prob) # 드롭아웃 레이어 (정규화)

    def forward(self, x):
        ## 입력 텐서 x를 받아 위치별로 통과시키고 출력 반환
        x = self.linear1(x) # 첫 번째 선형 레이어 통과 
        x = self.relu(x) # ReLU 적용
        x = self.dropout(x) # 드롭아웃 적용
        x = self.linear2(x) # 두 번째 선형 레이어 통과 
        return x