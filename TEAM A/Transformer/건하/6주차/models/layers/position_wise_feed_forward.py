from torch import nn
# 각 위치의 입력에 독립적으로 동일한 FeedForward 네트워크를 적용함 # Transformer 모델 내에서 매우 중요함 # 입력데이터의 비선형 변환 가능케함

class PositionwiseFeedForward(nn.Module): 
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 첫번째 선형 변환
        self.linear2 = nn.Linear(hiddem, d_model) # 2번째 선형 변환
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x): # 순전파 정의 
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x 