from torch import nn

class TokenEmbedding(nn.Embedding): # torch.nn을 사용한 토큰 임베딩
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1) # 패딩 토큰은 시퀀스의 길이를 맞추기 위해 사용됨