import math
from torch import nn

class ScaleDotProductAttention(nn.Module):
    '''
    Q: 주목하는 있는 문장 (Decoder)
    K: Query와의 관계를 확인할 모든 문장 (Encoder)
    V: Key와 동일한 문장들 (Encoder)
    '''
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) #마지막 차원에 적용 #초기화

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        # 1. Q와 K의 Trnaspose행렬과 Dot product을 통한 유사도 계산
        k_t = k.transpose(2,3) 
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. 선택적 마스킹 적용
        if mask is Not None:
            score = score.masked_fill(mask == 0, -1e4) # 마스킹 적용

        # 3. softmax함수를 적용하여 범위를 [0,1]로 변환
        score = self.softmax(score)

        # 4. 계산된 score를 이용해 V와 곱하기!
        v = score @ v 

        return v, score 

