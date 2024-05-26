import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder  # 인코더
        self.decoder = decoder  # 디코더 
        self.generator = generator  # 추가적인 FC layer(Embedding이 아닌 실제 target vocab에서의 token sequence)


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(self.tgt_embed(tgt), encoder_out, tgt_mask, src_tgt_mask)


    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        src_tgt_mask = self.make_src_tgt_mask(src, tgt)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_tgt_mask)
        # Decoder output의 마지막 dimension을 dembed에서 len(vocab)으로 변경
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)  # 각 vocabulary에 대한 확률값으로 변환, 성능 향상 목적
        return out, decoder_out
    
    # pad mask는 개념적으로 Encoder 내부에서 생성하는 것이 아니기 때문에
    # Transformer의 메소드로 위치함 
    # cross-attention인 경우, query는 source, key는 target
    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_tgt_mask(self, tgt):
        pad_mask = self.make_pad_mask(tgt, tgt)
        seq_mask = self.make_subsequent_mask(tgt, tgt)
        mask = pad_mask & seq_mask
        return pad_mask & seq_mask

    # Self-Multi-Head Attention Layer에서 넘어온 query, 
    # Encoder에서 넘어온 key, value 사이의 pad masking
    def make_src_tgt_mask(self, src, tgt):
        pad_mask = self.make_pad_mask(tgt, src)
        return pad_mask


    def make_pad_mask(self, query, key, pad_idx=1):
        # self_attention
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # -> embedding을 획득하기도 전, token sequence 상태로 들어오는 것 
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # pad_idx(1)와 일치하는 token들은 모두 0, 그 외에는 모두 1인 mask를 생성
        # 차원 확장 -> 어텐션 계산을 위한 브로드캐스팅 
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
        # 각 쿼리 포지션마다 모든 키 포지션에 대한 마스크를 적용할 수 있는 형태로 민든다 (질문)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask # 논리적 AND, 둘 다 패딩이 아닌 부분만 계산 
        mask.requires_grad = False
        return mask

    # i번째 token을 생성해낼 때, 1∼i−1의 token만 보이고, 
    # i+1∼의 token은 보이지 않도록 처리를 해야 하는 것
    def make_subsequent_mask(self, query, key):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device)
        return mask