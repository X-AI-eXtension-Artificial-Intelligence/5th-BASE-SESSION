import math
from torch import nn

### 3.2.1 Scaled Dot-Product Attention
class ScaleDotProductAttention(nn.Module):
    # Query: 디코더에서 집중하는 문장
    # Key: 쿼리와의 관계를 확인하기 위한 인코더의 모든 문장
    # Value: Key와 동일한 인코더의 모든 문장 
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # 어텐션 가중치 계산을 위한 소프트맥스 레이어

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size() # 입력 텐서의 차원 정보 얻기

        # 1. dot product Query with Key^T to compute similarity
        ## 쿼리와 키 전치 행렬의 dot product를 수행하여 유사도 점수 계산
        ## 차원 크기를 고려하여 d_tensor의 제곱근으로 스케일링
        ## 키 텐서 전치는 어텐션 계산 시 행렬 곱 연산 효율 향상을 위해 수행
        k_t = k.transpose(2, 3) # transpose
        score = (q @ k_t)/math.sqrt(d_tensor) # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            ## 패딩 요소는 스코어에서 매우 작은 값으로 채워 소프트맥스 후 거의 0에 가깝게 만들어 어텐션 방지
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        ## 모든 키 위치에 대한 합이 1이 되도록 어텐션 가중치 정규화
        score = self.softmax(score)

        # 4. multipy with Value
        v = score @ v
        
        return v, score
        # v: 가중치 합산된 값, score: 어텐션 가중치