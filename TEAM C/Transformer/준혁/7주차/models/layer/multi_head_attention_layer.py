import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc, dr_rate=0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        # deepcopy: 실제로는 서로 다른 weight를 갖고 별개로 운용되게 하기 위함
        # copy 없이 하나의 FC Layer로 Q, K, V를 모두 구하게 되면 항상 Q, K, V가 모두 같은 값일 것
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed) - attention 계산 이후 거쳐가는 FC Layer
        self.dropout = nn.Dropout(p=dr_rate)


    def calculate_attention(self, query, key, value, mask):
        # seq_len: 해당 mini-batch 내 token 개수의 최대 값
        # h: attention을 병렬적으로 h회 각각 수행
        # query, key, value: (n_batch, h, seq_len, d_k) -> 실제 model에 들어오는 input은 한 개의 문장이 아니라 mini-batch이기 때문에 
        # mask: (n_batch, seq_len, seq_len)
        d_k = key.shape[-1]
        ### Q와 K의 MatMul
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        ### Scale
        attention_score = attention_score / math.sqrt(d_k)  # scaling (gradient vanishing 방지)
        ### Mask
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
            # mask 텐서에서 0인 위치를 -1e9로 채운다.
            # 즉, 패딩된 부분의 어텐션 스코어를 매우 작은 값으로 설정하여, 이후에 softmax를 적용할 때 이 부분의 확률이 거의 0이 되도록 만든다. 
            # 이렇게 함으로써 모델이 패딩된 부분을 "무시"할 수 있도록 한다.
        ### SoftMax
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        ### MatMul
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out


    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed) -> 실제 Q, K, V가 아니라 input sentence embedding (n_batch×seq_len×d_embed) 
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        # FC Layer에 넣어서 Q, K, V값 구하기
        def transform(x, fc): # input: (n_batch, seq_len, d_embed)
            out = fc(x)       # output: (n_batch, seq_len, d_model) -> d_model = h * d_k
            # d_model을 h와 d_k로 분리하고, 각각을 하나의 dimension으로 분리
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            # calculate_attention의 input에 맞추기 위함 
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out
 
        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)       # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        # attention 계산
        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        # 다시 h와 d_k를 d_model로 결합
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        # FC Layer를 거쳐 d_model을 d_embed로 변환
        out = self.out_fc(out) # (n_batch, seq_len, d_embed) 
        return out