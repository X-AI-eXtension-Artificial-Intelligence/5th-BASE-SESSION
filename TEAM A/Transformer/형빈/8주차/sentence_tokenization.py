import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

english_file = './ko_en_data/english.txt'
korean_file = './ko_en_data/korean.txt'

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'



with open(english_file, 'r') as file:
    english_sentences = file.readlines()
with open(korean_file, 'r') as file:
    korean_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 100000
english_sentences = english_sentences[:TOTAL_SENTENCES]
korean_sentences = korean_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
korean_sentences = [sentence.rstrip('\n') for sentence in korean_sentences]

korean_vocabulary = []
for sen in korean_sentences:
    korean_vocabulary += list(sen)
    korean_vocabulary = list(set(korean_vocabulary))

english_vocabulary = []
for sen in english_sentences:
    english_vocabulary += list(sen)
    english_vocabulary = list(set(english_vocabulary))

korean_vocabulary.append('?')
korean_vocabulary.append(START_TOKEN)
korean_vocabulary.append(END_TOKEN)
korean_vocabulary.append(PADDING_TOKEN)
english_vocabulary.append('?')
english_vocabulary.append(START_TOKEN)
english_vocabulary.append(END_TOKEN)
english_vocabulary.append(PADDING_TOKEN)

index_to_korean = {k:v for k,v in enumerate(korean_vocabulary)}
korean_to_index = {v:k for k,v in enumerate(korean_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

max_sequence_length = 300
valid_sentence_indicies = []
for index in range(len(english_sentences)):
    english_sentence, english_sentence = english_sentences[index], english_sentences[index]
    if is_valid_length(english_sentence, max_sequence_length):
        valid_sentence_indicies.append(index)

korean_sentences = [korean_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]


class TextDataset(Dataset):

    def __init__(self, english_sentences, korean_sentences):
        self.english_sentences = english_sentences
        self.korean_sentences = korean_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.korean_sentences[idx]
    
dataset = TextDataset(english_sentences, korean_sentences)
 




