import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

## 데이터 로더를 구현하기
class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, pad_width=30):
        self.data_dir = data_dir
        self.transform = transform
        self.pad_width = pad_width

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        overlap_input = self.overlap_tile(input)

        overlap_input = overlap_input/255.0
        label = label/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if overlap_input.ndim == 2:
            overlap_input = overlap_input[:, :, np.newaxis]

        if self.transform:
            overlap_input = self.transform(overlap_input)
            label = self.transform(label)

        data = {'input': overlap_input, 'label': label}

        return data
    
    def overlap_tile(self, image):
        # 상하 부분을 반전하여 추출하고 원본 이미지에 붙임
        top = image[:self.pad_width, :][::-1, :]
        bottom = image[-self.pad_width:, :][::-1, :]
        padded_tb = np.concatenate([top, image, bottom], axis=0)

        # 좌우 부분을 반전하여 추출
        # 좌우 패딩을 추가하기 전에, padded_tb의 높이에 맞게 left와 right 패딩의 크기를 조정
        left = padded_tb[:, :self.pad_width][:, ::-1]
        right = padded_tb[:, -self.pad_width:][:, ::-1]

        # 좌우 패딩을 추가
        padded_complete = np.concatenate([left, padded_tb, right], axis=1)

        return padded_complete