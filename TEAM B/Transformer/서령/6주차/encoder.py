import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector

class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(params)  
        self.position_wise_ffn = PositionWiseFeedForward(params)  

    def forward(self, source, source_mask):
        # 입력 데이터와 마스크를 받아 처리

        # 레이어 정규화 후 self-attention을 수행함
        normalized_source = self.layer_norm(source)
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]

        # self-attention의 결과를 다시 정규화함
        normalized_output = self.layer_norm(output)

        # Position-Wise Feed-Forward network를 적용함
        output = output + self.position_wise_ffn(normalized_output)

        return output # 최종 출력 반환


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)  # token별로 embedding 구현
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)  # embedding 가중치를 초기화
        self.embedding_scale = params.hidden_dim ** 0.5  # embedding 결과에 적용할 scale 설정
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)  # 미리 계산된 Positional Encoding 적용
        self.encoder_layers = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])  # 여러 개의 Encoder layer를 생성
        self.dropout = nn.Dropout(params.dropout)  # dropout 적용
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)  # 최종 출력 정규화

    def forward(self, source):
        source_mask = create_source_mask(source)  # source sequence에 대한 마스크 생성
        source_pos = create_position_vector(source)  # source의 각 위치에 대한 position vector 생성

        source = self.token_embedding(source) * self.embedding_scale # source sequence embedding & scaling 적용
        source = self.dropout(source + self.pos_embedding(source_pos)) # 생성된 position vector에 Positional Encoding 적용하고 embedding된 source와 합쳐서 dropout 적용

        # 모든 Encoder layer를 순차적으로 실행
        for encoder_layer in self.encoder_layers:
            source = encoder_layer(source, source_mask)

        return self.layer_norm(source)  # 최종 출력을 정규화
