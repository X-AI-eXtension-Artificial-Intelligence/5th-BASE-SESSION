import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module): # 입력된 문장을 임베딩한다.
                    # Seq_Len : length of input sequence
                    # d_model : dimension of model == size of embedding vector
                    # vocab_size : size of vocabulary
    def __init__(self, d_model: int, vocab_size : int):
        super().__init_()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module): # 임베딩된 문장에 positional 인코딩을 해준다. 공식대로.
    
    def __init__(self, d_model : int, seq_len : int, dropout : float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        
        # creat a vector of shape (seq_len)
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # apply the sin to even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module): # 레이어 정규화를 정의. 
    
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module): # feedforward 블럭을 정의. linear1 -> relu -> dropout -> linear2
    
    def __init__(self, d_model : int, d_ff : int, dropout : float): # d_ff : feedforwar의 차원
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
        
    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model : int, h : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h" #* d_model % h == d_k
        
        self.d_k = d_model // h # d_v == d_k : 같은 길이임.
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask): # mask : 단어가 다른 단어와 상호 작용하지 않도록 하려는 경우 마스크를 사용.
        query = self.w_q(q) # == q' # (Batch, Seq_len, d_model) --> (Batch, Seq_Len, d_model)
        key = self.w_k(k) # == k' # (Batch, Seq_len, d_model) --> (Batch, Seq_Len, d_model)
        value = self.w_v(v) # == v' # (Batch, Seq_len, d_model) --> (Batch, Seq_Len, d_model)
        
        @staticmethod # 인스턴스 호출없이 부를 수 있음. # MultiheadAttentionBlock.attention으로 부를 수 있음.
        def attention(query, key, value, mask, dropout : nn.Dropout):
            d_k = query.shape[-1]
            # @ : 파이토치의 행렬곱 연산 --> mask --> softmax
            
            # (Batch, h, Seq_Len, d_k) -> (Batch, h, Seq_Len, Seq_Len)
            attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # sqrt(d_k)로 나누는 것은 논문상 그냥 안정화용도.
            if mask is not None: # mask가 정의되어 있다면.
                attention_scores.masked_fill(mask == 0, -1e9)
            attention_scores = attention_scores.sofmax(dim = -1) # (Batch, h, Seq_Len, Seq_Len)
            if dropout is not None:
                attention_scores = dropout(attention_scores)
            
            return (attention_scores @ value), attention_scores
        # for keeping batch dimension
        # multi head : head별로 입력의 다양한 부분에 attention할 수 있다.
        # 주제, 단어 사이의 의존성, 단어의 상대적 위치 등등
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, h, d_k) --> (Batch, h, Seq_Len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #! transpose 왜함?
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (Batch, h, Seq_Len, d_k) --> (Batch, Seq_Len, h, d_k) --> (Batch, Seq_Len, d_model)
        # 파이토치는 텐서의 모양을 변형하려면, 바로 view를 적용할 수 없어서 메모리를 연속적으로 배치하는 함수(?)를 사용해야 함.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k)
        
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        return self.w_o(x)
        
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # 실제로 논문에서는 self.norm(sublayer(x)) 순서임.
    

class EncoderBlock(nn.Module):
    # self attention? query, key, value의 값은 x 자체에서 나오므로 self라는 말이 붙음. 
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
        # sorce mask? 인코더의 입려겡 적용하려는 마스크
        def forward(self, x, src_mask):
            # lambda? sublayer의 표현
            x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask)) # self attention !!!
            x = self.residual_connections[1](x, self.feed_forward_block) # Feed forward
            return x

#! nn.ModuleList?
class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) #* 결국 가장 마지막 출력이 디코더로 전달될 것이다.
    

class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)]) #! nn.Module?
    # x : 디코더의 입력
    # src, tgt mask ? 번역 task이기 때문이다.
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, encoder_output, encoder_output))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, layers : nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask) # 디코더 블럭이므로, 괄호안의 값을 출력해야 함.
        return self.norm(x)
    
# 임베딩을 다시 단어의 위치로 변환할 linear layer가 필요 (= 어휘에 임베딩을 투영): projecting layer
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model : int, vocab_size : int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, Vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) #! dim = -1?
    

class Transformer(nn.Module):
    
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos : PositionalEncoding, projection_layer : ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.teg_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x): # 임베딩에서 vocab size를 가져온다.
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int = 512, N = 6, h : int = 8, dropout : float = 0.1, d_ff : int = 2048):
    # Create the Embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        return transformer