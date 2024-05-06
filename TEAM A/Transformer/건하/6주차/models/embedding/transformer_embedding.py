from torch import nn

from models.embedding.positional_encoding import PositionalEncoding
from models.embedding.token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model) # 토큰 임베딩 생성
        self.pos_emb = PositionalEncoding(d_model, max_len, device) # 포지션 인코딩 생성
        self.drop_out = nn.Dropout(p=drop_prob) # 드롭아웃 설정

    def forward(self, x):
        tok_emb = self_tok_emb(x) # 토큰 임베딩 적용
        pos_emb = self_pos_emb(x) # 포지션 인코딩 적용 
        return self.drop_out(tok_emb + pos_emb) # 결과를 드롭아웃 적용 후 변환