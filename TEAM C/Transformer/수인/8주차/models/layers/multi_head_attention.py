from torch import nn

from models.layers.scale_dot_product_attention import ScaleDotProductAttention

### 3.2.2 Multi-Head Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model) # 쿼리 투영 선형 변환층
        self.w_k = nn.Linear(d_model, d_model) # 키 투영 선형 변환층
        self.w_v = nn.Linear(d_model, d_model) # 값 투영 선형 변환층
        self.w_concat = nn.Linear(d_model, d_model) # 최종 출력 투영 선형 변환층

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        ## 입력 쿼리, 키, 값 텐서를 별도의 선형 변환층을 사용하여 투영
        ## 입력과 출력 차원은 동일 (d_model)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads (병렬 처리를 위해)
        ## 'split'함수를 사용하여 투영된 쿼리, 키, 값 텐서를 n_head 개의 부분 텐서로 분할
        ## 이를 통해모델은 정보에 병렬적으로 여러 각도에서 주의 집중 가능 ?
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        ## 각 헤드에 대해 scale dot prodcut attention 수행
        ## 유사도 점수 계산과 가중치 합산된 값 출력
        out, attention = self.attention(q, k, v, mask=mask) # 여기 어텐션은 ?

        # 4. concat and pass to linear layer
        ## 분할된 출력 텐서들을 다시 하나의 텐서로 연결
        out = self.concat(out)
        ## 연결된 출력 텐서를 마지막 선형 변환층(w_concat)을 통과시켜 출력 차원을 일치하게 조정
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out 

    def split(self, tensor):
        """
        텐서를 헤드 개수만큼 분할하는 함수

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head # 헤드당 차원 계산
        # 효율적인 분할을 위한 reshape와 transpose 수행
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        분할된 텐서들을 다시 연결하는 함수 
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor # 연결 후 층 차원

        # 전치, 모양 변경, 연결 
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor