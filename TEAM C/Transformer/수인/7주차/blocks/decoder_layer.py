from torch import nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # self attention 레이어 생성 (디코더 입력 벡터 간의 관계 파악)
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # layer norm 레이어 생성 (입력 값에 대한 통계량을 이용하여 정규화)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

        # encoder-decoder attention 레이어 (디코더 벡터와 인코더 벡터 간의 관계 파악)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # layer norm 레이어
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

        # 위치 정보를 고려한 feedforward 네트워크
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # layer norm 레이어 
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어 

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        ## 입력 벡터 간의 관계 파악하여 더 풍부한 표현 얻기 
        _x = dec # skip connection을 위한 백업 값 저장
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        ## 학습 안정성 향상과 gradient vanshing 완화
        x = self.dropout1(x)
        x = self.norm1(x + _x) # skip 연결 후 layer norm

        # 3. 인코더 벡터가 존재하는 경우 (encoder-decoder 어텐션 수행)
        if enc is not None:
            _x = x # skip connection을 위한 백업 값 저장
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            ## 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
            
            # 4. add and norm
            ## 학습 안정성 향상과 gradient vanshing 완화
            x = self.dropout2(x)
            x = self.norm2(x + _x) # skip 연결 후 layer norm

        # 5. positionwise feed forward network
        ## 순서 정보를 고려하여 추가적인 정보 추출
        _x = x # skip connection을 위한 백업 값 저장
        x = self.ffn(x)
        
        # 6. add and norm
        ## 학습 안정성 향상과 gradient vanshing 완화
        x = self.dropout3(x)
        x = self.norm3(x + _x) # skip 연결 후 layer norm
        return x