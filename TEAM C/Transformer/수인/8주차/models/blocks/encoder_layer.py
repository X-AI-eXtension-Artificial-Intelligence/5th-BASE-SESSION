from torch import nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # multihead attention 레이어
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        # layer normalization 레이어
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

        # posithionwise feed forward 네트워크 생성 
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        # layer normalization 레이어
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob) # 드롭아웃 레이어

    def forward(self, x, src_mask):
        # 1. compute self attention
        ## 입력 벡터 x에 대해 self attention 수행 후 문장 내 단어간 관계 파악
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        ## skip connection과 layer norm 수행 후 학습 안정성 향상
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        ## 위치 정보를 고려한 추가적인 정보 추출
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        ## 다시 한 번 skip connection과 layer norm 수행 
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x