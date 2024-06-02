import os
import numpy as np

import torch
import torch.nn as nn

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 데이터 디렉토리에 있는 모든 파일들의 리스트
        lst_data = os.listdir(self.data_dir)

        # label과 input 데이터 나눔
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        # 이 클래스의 파라미터로 지정
        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    # 실제로 데이터를 get하는 함수
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 데이터 normalize
        label = label/255.0
        input = input/255.0

        # 무슨 말이됴?
        if label.ndim == 2:
            label = label[:, :, np.newaxis]  # 라벨의 마지막 axis를 임의로 생성
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}  # 딕셔너리 형태로

        if self.transform:
            data = self.transform(data)

        return data
    
## 트랜스폼 구현하기
class ToTensor(object):  # numpy(Y, X, CH) -> tensor(CH, Y, X)
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)  # 넘파이의 채널을 첫번째로 옮기고 나머지는 그대로
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # 다시 딕셔너리로
        # 넘파이를 텐서로: from_numpy
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std  # label은 0 또는 1이기 때문에 적용 X

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)  # input과 lable은 항상 동시에 flip
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
