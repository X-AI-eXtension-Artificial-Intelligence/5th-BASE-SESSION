import pickle
import numpy as np
import torch
import torch.nn as nn

pickle_eng = open('pickles/eng.pickle', 'rb')
eng = pickle.load(pickle_eng)
pad_idx = eng.vocab.stoi['<pad>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weight(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.constant_(layer.bias,0)

def create_positional_encoding(max_len, hidden_dim):
    sinusoid_table = np.array([pos/np.power(10000,2*i/hidden_dim) for pos in range(max_len) for i in range(hidden_dim)])
    sinusoid_table = sinusoid_table.reshape(max_len, -1)

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    sinusoid_table = torch.FloatTensor(sinusoid_table).to(device)
    sinusoid_table[0] = 0.

    return sinusoid_table

def create_source_mask(source):
    source_length = source.length

    source_mask = (source == pad_idx) # pad가 있으면 masking
    source_mask = source_mask.unsqueeze(1).repeat(1, source_length ,1)

    return source_mask

def create_position_vector(sentencce):
    batch_size = sentencce.size()
    pos_vec = np.array([(pos+1) if word != pad_idx else 0 for row in range(batch_size) for pos,word in enumerate (sentencce[row])])  # pad를 0으로 나머지 1씩 미룸
    pos_vec - pos_vec.reshape(batch_size,-1)
    pos_vec = torch.LongTensor(pos_vec).to(device)
    return pos_vec

def create_subsequent_mask(target):
    batch_size, target_length = target.size()

    subsequent_mask = torch.triu(torch.ones(target_length, target_length), diagonal=1).bool().to(device)
    subsequent_mask = subsequent_mask.unsqueeze(0).repeat(batch_size,1,1)

    return subsequent_mask

def create_target_mask(source, target):
    target_length = target.shape[1]
    subsequent_mask = create_subsequent_mask(target)
    source_mask = (source == pad_idx)
    target_mask = (target == pad_idx)

    dec_enc_mask = source_mask.unsqueeze(1).repeat(1, target_length, 1)
    target_mask = target_mask.unsqueeze(1).repeat(1, target_mask, 1)

    target_mask = target_mask | subsequent_mask

    return target_mask, dec_enc_mask