import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.ops import init_weight

class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super(MultiHeadAttention, self).__init__()
        # 각 head가 처리할 벡터의 차원을 동일하게 유지하기 위해 hidden_dim은 n_head로 나누어 떨어지는지 차원을 검증함
        assert params.hidden_dim % params.n_head == 0
        # 각 head가 독립적으로 입력 데이터에 대해 attention을 수행할 수 있도록 n_head의 수만큼 SelfAttention 모듈을 생성함
        self.attentions = nn.ModuleList([SelfAttention(params) for _ in range(params.n_head)])
        # 각 head에서 계산된 결과를 다시 원래 차원으로 매핑하기 위한 선형 변환을 정의함
        self.o_w = nn.Linear(params.hidden_dim, params.hidden_dim, bias=False)
        init_weight(self.o_w)  # 위에서 정의한 선형 변환의 가중치 초기화 
        self.dropout = nn.Dropout(params.dropout) # dropout 적용

    def forward(self, query, key, value, mask=None):
        # 입력으로 받은 query, key, value를 각 SelfAttention 모듈에 전달하여 attention을 수행함
        # mask가 제공되면, 특정 위치를 마스킹 처리하여 계산에서 제외함
        self_attentions = [attention(query, key, value, mask) for attention in self.attentions]
        # 각 head의 attention 결과를 수집함
        weighted_vs = [weighted_v[0] for weighted_v in self_attentions] # 모든 head의 출력값을 담고 있는 리스트
        attentions = [weighted_v[1] for weighted_v in self_attentions] # 모든 head의 attention score를 담고 있는 리스트

        # 모든 head의 결과를 연결함
        weighted_v = torch.cat(weighted_vs, dim=-1)

        # 연결된 결과에 선형 변환을 적용하고 드롭아웃을 통해 최종 출력을 생성함
        output = self.dropout(self.o_w(weighted_v))

        return output, attentions  # 최종 출력과 각 head의 attention score 반환함


class SelfAttention(nn.Module):
    def __init__(self, params):
        super(SelfAttention, self).__init__()
        self.hidden_dim = params.hidden_dim
        self.attention_dim = params.hidden_dim // params.n_head  # 각 head에서 처리할 차원

        # query, key, value를 위한 선형 변환을 정의함
        self.q_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.k_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        self.v_w = nn.Linear(self.hidden_dim, self.attention_dim, bias=False)
        # 선형 변환의 가중치 초기화
        init_weight(self.q_w) 
        init_weight(self.k_w)
        init_weight(self.v_w)

        self.dropout = nn.Dropout(params.dropout)  # dropout 적용
        # scaling factor를 계산함 -> 차원(attention_dim)의 제곱근
        self.scale_factor = torch.sqrt(torch.FloatTensor([self.attention_dim])).to(params.device)

    def forward(self, query, key, value, mask=None):
        # 입력된 query, key, value에 각각의 선형 변환을 적용하여 Q, K, V 행렬을 생성함
        q = self.q_w(query)
        k = self.k_w(key)
        v = self.v_w(value)

        # Q와 K의 행렬 곱(dot product)을 계산한 후, scaling factor로 나누어 정규화 -> attention을 수행함
        self_attention = torch.bmm(q, k.permute(0, 2, 1)) / self.scale_factor

        # mask가 제공되면, 특정 위치를 마스킹 처리하여 계산에서 제외함
        # 이 때,마스크가 적용된 위치는 -np.inf로 설정하여 소프트맥스 적용 시 영향력을 제거함
        if mask is not None:
            self_attention = self_attention.masked_fill(mask, -np.inf)

        # 소프트맥스 함수를 이용하여 attention_score를 계산함
        attention_score = F.softmax(self_attention, dim=-1)
        # dropout을 거쳐 attention score가 정규화됨
        norm_attention_score = self.dropout(attention_score)

        # 정규화된 attention score를 사용하여 V 행렬에 가중 평균을 적용함
        weighted_v = torch.bmm(norm_attention_score, v)

        return self.dropout(weighted_v), attention_score  # dropout 적용된 결과와 attention score 반환