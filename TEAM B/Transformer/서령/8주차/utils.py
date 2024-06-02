import os
import re
import json
import pickle
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm

from torchtext import data as ttd
from torchtext.data import Example, Dataset


def load_dataset(mode):
    print(f'Loading AI Hub Kor-Eng translation dataset and converting it to pandas DataFrame . . .')

    data_dir = Path().cwd() / 'data'

    if mode == 'train':
        train_file = os.path.join(data_dir, 'train.csv')
        train_data = pd.read_csv(train_file, encoding='utf-8')

        valid_file = os.path.join(data_dir, 'valid.csv')
        valid_data = pd.read_csv(valid_file, encoding='utf-8')

        print(f'Number of training examples: {len(train_data)}')
        print(f'Number of validation examples: {len(valid_data)}')

        return train_data, valid_data

    else:
        test_file = os.path.join(data_dir, 'test.csv')
        test_data = pd.read_csv(test_file, encoding='utf-8')

        print(f'Number of testing examples: {len(test_data)}')

        return test_data


def clean_text(text):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
    return text


def convert_to_dataset(data, kor, eng):
    missing_rows = [idx for idx, row in data.iterrows() if type(row.korean) != str or type(row.english) != str]
    data = data.drop(missing_rows)

    list_of_examples = [Example.fromlist(row.apply(lambda x: clean_text(x)).tolist(),
                                         fields=[('kor', kor), ('eng', eng)]) for _, row in data.iterrows()]

    dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])

    return dataset


def make_iter(batch_size, mode, train_data=None, valid_data=None, test_data=None):
    file_kor = open('pickles/kor.pickle', 'rb')
    kor = pickle.load(file_kor)

    file_eng = open('pickles/eng.pickle', 'rb')
    eng = pickle.load(file_eng)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mode == 'train':
        train_data = convert_to_dataset(train_data, kor, eng)
        valid_data = convert_to_dataset(valid_data, kor, eng)

        print(f'Make Iterators for training . . .')
        train_iter, valid_iter = ttd.BucketIterator.splits(
            (train_data, valid_data),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return train_iter, valid_iter

    else:
        test_data = convert_to_dataset(test_data, kor, eng)

        dummy = list()

        print(f'Make Iterators for testing . . .')
        test_iter, _ = ttd.BucketIterator.splits(
            (test_data, dummy),
            sort_key=lambda sent: len(sent.kor),
            sort_within_batch=True,
            batch_size=batch_size,
            device=device)

        return test_iter


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def display_attention(candidate, translation, attention):
    attention = attention.cpu().detach().numpy()

    font_location = 'pickles/NanumSquareR.ttf'
    fontprop = fm.FontProperties(fname=font_location)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + [t.lower() for t in candidate], rotation=45, fontproperties=fontprop)
    ax.set_yticklabels([''] + translation, fontproperties=fontprop)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


class Params:
    def __init__(self, json_path):
        self.update(json_path)
        self.load_vocab()

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def load_vocab(self):
        pickle_kor = open('pickles/kor.pickle', 'rb')
        kor = pickle.load(pickle_kor)

        pickle_eng = open('pickles/eng.pickle', 'rb')
        eng = pickle.load(pickle_eng)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        params = {'input_dim': len(kor.vocab), 'output_dim': len(eng.vocab),
                  'sos_idx': eng.vocab.stoi['<sos>'], 'eos_idx': eng.vocab.stoi['<eos>'],
                  'pad_idx': eng.vocab.stoi['<pad>'], 'device': device}

        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__
