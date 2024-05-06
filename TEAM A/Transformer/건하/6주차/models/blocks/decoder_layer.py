from torch import nn 

from models.layers.layer_norm import LayerNorm
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module): # nn.Modue 클래스는 pytorch에서 제공하는 클래스임
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__() # 부모 클래스(DecoderLayer)의 생성자를(nn.Moduel) 호출함 # DecoderLayer 클래스가 nn.Moduel의 모든 기능을 제대로 상속받고 초기화되어 사용될 수 있도록 하는 과정
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head) # Encoder-Decoder Attention을 구현하기 위해 사용 # Decoder가 Encoder의 정보를 활용하여 보다 정확한 출력을 생성하도록 하는 요소
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. self attention
        _x = dec 
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add & norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. Encoder-Decoder Attention 계산
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask) # 인코더와 출력과의 관계를 파악하기 위해 수행됨
            # Query는 현재 Decoder의 입력을 의미 # key와 value는 모두 Encoder의 출력을 사용 -> 이때 인코더의 출력은 입력 시퀀스를 처리한 결과로, 각 입력 단어 or 토큰에 대한 정보를 담고 있음
            ## key를 사용하여 각 인코더 출력의 요소가 현재 디코더가 처리하고 있는 Query와 얼마나 관련 있는지를 평가함
            ### Attention score는 Value에 적용되어, 어떤 인코더의 출력을 디코더의 다음 단계로 전달할지를 결정함 
            #### 즉, 디코더의 현재 상태(Query)와 인코더의 출력(Key, Value) 사이의 관계를 계산하고, 이를 통해 Decoder가 어떤 Encoder 출력에 주목해야 할지를 결정함 / 마스크는 특정 부분을 무시하도록 하여, 모델이 불필요하거나 잘못된 정보에 주목하는 것을 방지함

            # 4. add & norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)
        
        # 5. positionwise ffn
        _x = x 
        x = self.ffn(x)

        # 6. add & norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x 
