import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_subsequent_mask(target):
    # target sequence의 길이를 기반으로 subsequent mask 생성 -> 각 위치에서 자신보다 뒤에 오는 위치의 정보에 접근하지 못하도록 함
    batch_size, target_length = target.size()
    # torch.triu는 대각선(diagonal) 위의 요소는 1, 그 아래는 0인 상삼각행렬을 생성힘
    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return subsequent_mask

def create_source_mask(source):
    # source mask를 생성함 -> source sequence에서 padding 부분을 마스킹함
    source_length = source.shape[1]
    source_mask = (source == pad_idx)
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length, 1)
    return source_mask

def create_target_mask(source, target):
    # target mask와 Decoder-Encoder mask를 생성함
    target_length = target.shape[1]
    subsequent_mask = create_subsequent_mask(target)
    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)

    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_length, 1)

    target_mask = target_mask | subsequent_mask
    return target_mask, dec_enc_mask

def create_position_vector(sentence):
    # position vector를 생성함 -> 문장 내 각 단어의 위치를 나타내는 벡터를 생성하는것
    batch_size, _ = sentence.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0
                        for row in range(batch_size) for pos, word in enumerate(sentence[row])])
    pos_vec = pos_vec.reshape(batch_size, -1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec

def create_positional_encoding(max_len, hidden_dim):
    # 문장의 각 위치에 대해 사인 함수과 코사인 함수를 사용하여 positional encoding을 생성함
    sinusoid_table = np.array([pos / np.power(10000, 2 * i / hidden_dim)
                               for pos in range(max_len) for i in range(hidden_dim)])
    sinusoid_table = sinusoid_table.reshape(max_len, -1)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.
    return sinusoid_table

def init_weight(layer):
    # Xavier 초기화 방식으로 가중치 초기화 
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
