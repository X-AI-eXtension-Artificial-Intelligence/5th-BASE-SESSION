import torch
import torch.nn as nn 
import math

#Input embedding 
class InputEmbeddings(nn.Module):

    def __init__(self,d_model : int, vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model) 
     

#Positional Encoding
class PositionalEncoding(nn.Module):
    

    def __init__(self, d_model : int, seq_len: int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)

    
        position = torch.arange(0,seq_len, dtype = torch.float).unsqueeze(1)

        _2i = torch.exp(torch.arange(0,d_model,2,dtype=torch.float))

        pe[:,0::2] = torch.sin(position/10000**(_2i/d_model)) 
        pe[:,1::2] = torch.cos(position/10000**(_2i/d_model))

        #차원 늘리기
        pe = pe.unsqueeze(0) #(1,Seq_len,d_model)

        self.register_buffer('pe',pe)
    
    def forward(self,x):  # input
        # input + positional encoding
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
   
   
#레이어 정규화   
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None : 
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        
    def forward(self,x):
        mean = x.mean(dim =-1, keepdim=True)
        std = x.std(dim= -1, keepdim =True)
        return self.alpha * (x-mean) / (std+ self.eps) + self.bias
    

#FeedForward
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model : int, d_ff : int, dropout : float) -> None:
        super().__init__()
        # feed forward upwards projection size
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) #차원을 다시 512차원으로
        
    def forward(self,x):
        
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))      


#Multi-Head Attention
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self,d_model : int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # head 개수
        
        assert d_model % h ==0, 'd_model is not divisible by h'      
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model,d_model) #Wk
        self.w_v = nn.Linear(d_model,d_model) #Wv
        
        self.w_o = nn.Linear(d_model,d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1] # (Batch, seq_len, d_model)
        
        #(Batch, h, Seq_len, d_k) -> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
 
        attention_scores = attention_scores.softmax(dim=-1) #(Batch,h,seq_len, seq_len)

        if dropout is not None : 
            attention_scores = dropout(attention_scores)
        #최종적으로 어텐션 가중합된 결과와 어텐션 가중치를 반환
        return (attention_scores @ value), attention_scores
    
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        
        #Q,K,V를 head 수 만큼 분리
        #(Batch, seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # 전치
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        #(Batch, h, Seq_len, d_k) -> (Batch, Seq_len, h,  d_k) -> (Batch,Seq_len, d_k)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k) 
        
        #(Batch,Seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)
    

#ResidualConnection
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    #sublayer ?    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    #ProjectionLayer
class ProjectionLayer(nn.Module):
    
    def __init__(self,d_model : int, vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        #(Batch, seq_len,d_model) -> (Batch,seq_len,vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)