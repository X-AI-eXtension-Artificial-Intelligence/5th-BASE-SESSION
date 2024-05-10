import torch
import torch.nn as nn
import math

## Input Embedding / 입력 문장 -> 512차원의 vector
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model # dimension
        self.vocab_size = vocab_size # vocabulary 단어 개수
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # 논문에서 설명한 weights 곱해줌
        return self.embedding(x) * math.sqrt(self.d_model) 

## Positional Encoding / Information about the position of each word inside the sentence
    # 엠베딩과 같은 사이즈의 벡터(512)    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # 512 벡터 사이즈
        self.seq_len = seq_len # 문장의 maximun len
        self.dropout = nn.Dropout(dropout)
        
        # matrix of shape (seq_len, d_model) 문장의 단어 수*벡터사이즈
        pe = torch.zeros(seq_len, d_model)
        # Vector (seq_len, 1) 생성
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # 짝수 position에 sine 적용
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        
        # 홀수 position에 cosine 적용
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        
        # batch dimension / 배치에 있는 모든 문장에 적용
        pe = pe.unsqueeze(0) # (1(배치 차원), seq_len, d_model)
        # Register the positional encoding as a buffer, 모델에 함께 저장
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        # 이 텐서는 학습하는 동안 변하지 않는 값이므로 false
        # (batch, seq_len, d_model)
        return self.dropout(x)


## Feed Forward /  Fully Connected layer
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size(512)
        self.h = h # Number of heads, d_model이 h로 나누어떨어질 수 있어야함
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # 각 head의 Dimension of vector
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq / (d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv 모두 같은 값 
        
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo / (h*dv, d_model)
        self.dropout = nn.Dropout(dropout)

    # attention 계산하는 메서드
    @staticmethod #instance없이 .으로 호출 가능
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # 논문 수식 구현
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 매우 낮은값으로 replace (indicating -inf) -> softmax지나서 0됨
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return tuple, for visualization
        return (attention_scores @ value), attention_scores
    
    # Forward
    def forward(self, q, k, v, mask): 
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

        # 더 작은 metrics로 분리하여 각 head에 할당
        # 문장이 아니라 엠베딩을 split하고싶은 것 -> batch dimension([0])은 keep
        # 두번째 dimensiond인 sequence도 split하지x -> keep([1])
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        # transpose: 각 head에서 seq_len, d_k 볼 수 있도록(full sentence) 
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # attention 메서드 이용해서 계산
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)
    
## layer normalization 
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps # numerical stability 위함 / sigma값이 0에 가까워져서 분모가 0 되는걸 막기위해 + epsilon해줌
        self.alpha = nn.Parameter(torch.ones(features)) #will be multiplied / nn.: makes the parameter learnable
        self.bias = nn.Parameter(torch.zeros(features)) # will be added

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # 논문상의 수식 구현
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

## skip connection부분
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer): #sublayer: output of the next layer
            return x + self.dropout(sublayer(self.norm(x)))
        #논문상에선sublayer 다음에 normalization 적용

###############################
## encoder & decoder
 # encoder block
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):#incoder input에 적용하는 source mask: padding word를 다른 단어들과 interact못하도록
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) #1st skip connection,encoder에서는 xxx같은 들어 값 들어감
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# Encoder / 인코더블록*n   
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # n개 반복
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


## Decoder Block
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block # key, value는 encoder에서, query는 decoder에서 연결
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)]) #3개 connection

    def forward(self, x, encoder_output, src_mask, tgt_mask): # x: decoder input, src_mask: encoder에서 오는 mask(eng), target mask: decoder target language mask, 3개 레이어 build
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

#Decoder build
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

# decoder 지나서 linear layer, (seq,d_model) embedding을 다시 position of the vocabulary 
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # d_model -> vocab_size

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) # self.proj(x)


# Transformer architecture   
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # source language에 대한 embedding
        self.tgt_embed = tgt_embed # target language에 대한 embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src) #embedding
        src = self.src_pos(src) #positional encoding
        return self.encoder(src, src_mask) #encoder통과
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt) #embedding
        tgt = self.tgt_pos(tgt) #positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask) #decode
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
# Build Transformer    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer: # d_ff: feed forward layer의 hidden layer
    # Create the embedding layers / token of the vocabulary -> vector(size 512)
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N): # encoder block N개 진행
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks)) # N개 블록 
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) #target vocabulary로 proj
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # 빠른학습가능하게 파라미터 초기화
    
    return transformer






