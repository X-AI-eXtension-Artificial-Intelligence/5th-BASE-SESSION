import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positionwise import PositionWiseFeedForward
from model.ops import create_positional_encoding, create_source_mask, create_position_vector

class EncoderLayer(nn.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps = 1e-6)
        self.self_attention = MultiHeadAttention(params)
        self.position_wise_ffn = PositionWiseFeedForward(params)

    def forward(self, source, source_mask):
        normalized_source = self.layer_norm(source)
        output = source + self.self_attention(normalized_source, normalized_source, normalized_source, source_mask)[0]

        normalized_output = self.layer_norm(output)
        output = output + self.position_wise_ffn(normalized_output)

        return output
    

class Encoder(nn.Module):
    def __init__(self,params):
        super(Encoder,self).__init__()
        self.token_embedding = nn.Embedding(params.input_dim, params.hidden_dim, padding_idx=params.pad_idx)

        nn.init.normal_(self.token_embedding,mean=0,std=params.hidden_dim**-0.5)
        self.embedding_scale = params.hidden_dim**0.5

        self.pos_embedding = nn.Embedding.from_pretrained(create_positional_encoding(params.max_len+1,params.hidden_dim),freeze=True)

        self.encoder_layer = nn.ModuleList([EncoderLayer(params) for _ in range(params.n_layer)])
        self.dropout = nn.Dropout(params.dropout)
        self.layer_norm = nn.LayerNorm(params.hidden_dim, eps=1e-6)

    def forward(self,source):

        source_mask = create_source_mask(source)
        source_pos = create_position_vector(source)

        source = self.token_embedding(source)*self.embedding_scale
        source = self.dropout(source+self.pos_embedding(source_pos))

        for encoder_layer in self.encoder_layer:
            source = encoder_layer(source,source_mask)
        
        return self.layer_norm(source)