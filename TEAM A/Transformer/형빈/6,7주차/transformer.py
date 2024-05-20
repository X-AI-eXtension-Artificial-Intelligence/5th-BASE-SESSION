import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from sentence_tokenization import *

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#####################
# by scaling -> make rower variance -> easier for training
def scaled_dot_product(q,k,v, mask=None):
    # q, k, v each = 30 x 8 x 200 x 64
    d_k = q.size()[-1] # 64
    scaled = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k) #transpose(-1,-2)는 마지막 두개의 위치만 바꿈
    # 30 x 8 x 200 x 200
    if mask is not None:
        scaled += mask # 30 x 8 x 200 x 200 + 200 x 200

    attention = F.softmax(scaled,dim=-1) # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v) # values means all context information of words
    # 30 x 8 x 200 x 64
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0,self.d_model,2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length,1))
        even_PE = torch.sin(position/denominator)
        odd_PE = torch.cos(position/ denominator)
        stacked = torch.stack([even_PE,odd_PE],dim=2) # cat은 그냥 합침, stack은 dim으로 합침
        PE = torch.flatten(stacked,start_dim=1,end_dim=2)
        return PE

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model) # 512 * 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 * 512

    def forward(self, x, mask= None):
        batch_size, max_sequence_length, d_model = x.size() # 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536 # q,k,v concat
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0,2,1,3) # 30 x 8 x 200 x 192 # permute == transpose
        q, k, v = qkv.chunk(3, dim=-1) # each are 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) # attention = 30 x 8 x 200 x 200  # value 30 x 8 x 200 x 64
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 512
        out = self.linear_layer(values)
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model// num_heads
        self.kv_layer = nn.Linear(d_model,2*d_model)
        self.q_layer = nn.Linear(d_model,d_model)
        self.linear_layer = nn.Linear(d_model,d_model)

    def forward(self,x,y,mask=None):
        batch_size = sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size,sequence_length,self.num_heads, 2* self.head_dim)
        q = q.reshape(batch_size,sequence_length,self.num_heads,self.head_dim)

        kv = kv.permute(0,2,1,3)
        q = q.permute(0,2,1,3)
        k, v = kv.chunk(2,dim=-1)
        values , attention = scaled_dot_product(q,k,v,mask)
        values = values.reshape(batch_size,sequence_length,d_model)
        out = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self,parameters_shape,eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape # 512
        self.eps = eps # prevent inf at std calculating
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # 512 # std
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # 512 # mean

    def forward(self, inputs): # 30 x 200 x 512
        dims = [-(i + 1) for i in range(len(self.parameters_shape))] # [-1] 
        mean = inputs.mean(dim = dims, keepdim = True) # 30 x 200 x 1
        var = ((inputs-mean) ** 2).mean(dim=dims,keepdim = True) # 30 x 200 x 1
        std = (var + self.eps).sqrt() # 30 x 200 x 1
        y = (inputs - mean) / std # 30 x 200 x 512
        out = self.gamma * y + self.beta  # (30 x 200 x 1) x 512
        # gamma and beta is learnable parameters
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 x 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x): # 30 x 200 x 512
        x = self.linear1(x) # 30 x 200 x 2048
        x = self.relu(x) # 30 x 200 x 2048
        x = self.dropout(x) # 30 x 200 x 2048
        x = self.linear2(x) # 30 x 200 x 512
        return x

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x



class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super(EncoderLayer,self).__init__() ## 왜 super에 자식 클래스 인자로 넣었지? init 값도 지정 안해줬는디
        self.attention = MultiHeadAttention(d_model=d_model, num_heads = num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual_x = x # 30 x 200 x 512
        x = self.attention(x,mask=self_attention_mask) # 30 x 200 x 512
        x = self.dropout1(x) # 30 x 200 x 512
        x = self.norm1(x + residual_x) # 30 x 200 x 512
        residual_x = x # 30 x 200 x 512
        x = self.ffn(x) # 30 x 200 x 512
        x = self.dropout2(x) # 30 x 200 x 512
        x = self.norm2(x + residual_x) # 30 x 200 x 512
        return x
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y

class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x


##########################
# perform encoding for num_layers with defined encoder layer
# class Encoder(nn.Module):
#     def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers):
#         super().__init__()
#         self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
#                                       for _ in range(num_layers)])
        
#     def forward(self,x):
#         x - self.layers(x)
#         return x # final output after 5 layers of encoder is capture the context
    

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

# class SequentialDecoder(nn.Sequential): 
#     def forward(self, *inputs):
#         x,y,mask = inputs
#         for module in self.modules.values():
#             # 번역에 사용되는 디코더 단계마다 새로운 단어를 추출하기 위해 생성
#             # x 는 인코더 단계를 모두 마친 벡터이므로, 변경할 필요 없음.
#             y = module(x,y,mask) # 30 x 200 x 512 

#         return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y


# class Decoder(nn.Module):
#     def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers=1):
#         super().__init__()
#         self.layers = SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob) 
#                                           for _ in range(num_layers)])
        
#     def forward(self,x,y,mask):
#         # x : 30 x 200 x 512
#         # y : 30 x 200 x 512
#         # mask : 200 x 200
#         y = self.layers(x,y,mask)
#         return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                kn_vocab_size,
                english_to_index,
                korean_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, korean_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out