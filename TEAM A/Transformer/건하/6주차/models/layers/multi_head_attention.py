from torch import nn

from models.layer.scale_dot_product_attention import ScaleDotProductAttention

# 입력 데이터에 대한 Attention을 여러 Head를 통해 독립적으로 계산하고 그 결과를 결합하는 과정 
class MultiHeadAttention(nn.Module):
    def__init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model) 

    def forward(self, q, k, v, mask=None):

        # 1. 가중치 행렬과의 Dot product
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. Head 수에 따른 Tensor Split
        q, k ,v = self.split(q), self.split(k), self.split(v)

        # 3. ScaleDotProduct Attention으로 유사도 계산
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 결과를 결합하고 선형 층 통과
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. Attention map 시각화
        # 코드 없음
        
        return out
    
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head # 모델의 차원(512)과 헤드 수(8)이면, 각 헤드가 64차원의 데이터를 처리한다는 의미임 (d_tensor) -> 헤드 당 차원수 의미
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1,2) # [batch_size, self.n_head, length, d_tensor]로 변환
        # view -> 텐서 차원 재배열  # transpose -> 차원 순서 교환  
        # 이러한 변형을 통해 multiheadattention은 입력 데이터를 head 수만큼 분할하여 각 head가 서로 다른 정보를 추출할 수 있게 하며 다양한 관점에서 정보를 학습할 수 있게 도와줌 + 나중에 다시 결합하여 더 풍부한 표현가능

        return tensor

    def concat(self, tensor):
        bath_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, d_model) 
        # 다시 원래의 형태로 병합하는 과정을 통해 각 시퀀스 위치에 대해 모든 헤드의 출력을 연속적으로 배열하는 데 도움을 줌
        # transpose 연산 수행한 후에는 메모리 내에서 텐서의 데이터가 연속적이지 않게 될 수도 있음. 그래서
        # contiguous -> 텐서의 데이터를 메모리에서 연속적인 형태로 재배열함 -> # view를 사용 하기 전 텐서의 형태를 안전하게 변경하기 위해 필요함

        # 각 위치의 헤드 출력들이 하나의 벡터로 합쳐지고 최종적으로 입력 데이터의 차원 수와 일치하게 됨 

        # 이렇게 텐서를 재구성하는 이유는 멀티헤드어텐션에서 각 헤드가 독립적으로 계산한 정보를 다시 하나의 표현으로 통합하여 이후의 네트워크 레이어들이 이 정보를 활용할 수 있도록 만들기 위함


        return tensor 