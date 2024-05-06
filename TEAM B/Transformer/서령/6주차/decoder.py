import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_target_mask, create_position_vector

class DecoderLayer(nn.Module):
    def __init__(self, params):
        super(DecoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)  
        self.self_attention = MultiHeadAttention(params)  
        self.encoder_attention = MultiHeadAttention(params)  
        self.position_wise_ffn = PositionWiseFeedForward(params)  

    def forward(self, target, encoder_output, target_mask, dec_enc_mask):

        # 레이어 정규화 후 self-attention을 수행함
        norm_target = self.layer_norm(target)
        output = target + self.self_attention(norm_target, norm_target, norm_target, target_mask)[0]

        # self-attention의 결과를 다시 정규화함
        norm_output = self.layer_norm(output)
        # Encoder attention 수행
        sub_layer, attn_map = self.encoder_attention(norm_output, encoder_output, encoder_output, dec_enc_mask)
        output = output + sub_layer

        # 다시 정규화 한 다음, Position-Wise Feed-Forward network를 적용함
        norm_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(norm_output)

        return output, attn_map  # 최종 출력과 attention map 반환

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.token_embedding = nn.Embedding(params.output_dim, params.hidden_dim, padding_idx=params.pad_idx)  # token별로 embedding 구현
        nn.init.normal_(self.token_embedding.weight, mean=0, std=params.hidden_dim**-0.5)  # embedding 가중치 초기화
        self.embedding_scale = params.hidden_dim ** 0.5  # embedding 결과에 적용할 scale 설정
        self.pos_embedding = nn.Embedding.from_pretrained(
            create_positional_encoding(params.max_len+1, params.hidden_dim), freeze=True)  # 미리 계산된 Positional Encoding 적용
        self.decoder_layers = nn.ModuleList([DecoderLayer(params) for _ in range(params.n_layer)])  # 여러 개의 Decoder layer를 생성
        self.dropout = nn.Dropout(params.dropout)  # dropout 적용
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6) 

    def forward(self, target, source, encoder_output):
        target_mask, dec_enc_mask = create_target_mask(source, target)  # target mask 및 decoder-encoder 간의 mask를 생성함
        target_pos = create_position_vector(target)  # target의 각 위치에 대한 position vector 생성

        target = self.token_embedding(target) * self.embedding_scale  # target에 token embedding과 scaling 적용
        target = self.dropout(target + self.pos_embedding(target_pos))  # 생성된 position vector에 Positional Encoding 적용하고 embedding된 target과 합쳐서 dropout 적용

        for decoder_layer in self.decoder_layers:
            target, attention_map = decoder_layer(target, encoder_output, target_mask, dec_enc_mask)  # 각 Decoder layer을 순차적으로 실행

        target = self.layer_norm(target)  # 최종 정규화
        output = torch.matmul(target, self.token_embedding.weight.transpose(0, 1))  # 최종 출력 계산

        return output, attention_map  # 최종 출력과 attention map 반환
