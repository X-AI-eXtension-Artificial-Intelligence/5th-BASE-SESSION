from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # 멀티헤드어텐션
        self.norm1 = LayerNorm(d_model=d_model) # 첫번째 레이어 정규화
        self.dropout1 = nn.Dropout(p=drop_prob) # 첫번째 드롭아웃 

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob) # 포지션별 FFN
        self.norm2 = LayerNorm(d_model=d_model) # 두번째 레이어 정규화
        self.dropout2 = nn.Dropout(p=drop_prob) # 두번째 드롭아웃

    def forward(self, x, scr_mask):

        # 1. self attention 계산
        _x = x # residual connection을 위함 # 원래의 입력을 더해줌으로써 각 층의 출력에 입력의 정보가 유지되고, 전체 네트워크의 학습 안정성 및 성능 향상
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward networks
        _x = x # 중간 결과 저장
        x = self.ffn(x)

        # 4. add & norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x 

