import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms

# tensor shape (1, x, y)
def mirroring_Extrapolate(img):
    # mirroring 92 pixel

    x = img.shape[1]
    y = img.shape[2]

    np_img = np.array(img)

    np_img = np_img[0]

    if x < 388:
        pad_x_left = (572 - x) / 2
        pad_x_right = (572 - x) / 2
    else:
        pad_x_left = 92
        pad_x_right = 388 - (x % 388) + 92

    if y < 388:
        pad_y_up = (572 - y) / 2
        pad_y_down = (572 - y) / 2
    else:
        pad_y_up = 92
        pad_y_down = 388 - (y % 388) + 92

    np_img = np.pad(np_img, ((pad_x_left, pad_x_right), (pad_y_up, pad_y_down)), 'reflect')

    np_img = np_img[:, :, np.newaxis]

    return torch.from_numpy(np_img.transpose((2, 0, 1)))

## 데이터 로더 구현하기 
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, mode='Original'):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)  # File list

        # Filter lists
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        # Access label and input data using indexing
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # Normalize
        label = label / 255.0
        input = input / 255.0

        # Add channel dimension if necessary
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # Apply mode-specific preprocessing (if applicable)
        if self.mode == 'Mirroring':
            img = mirroring_Extrapolate(img)  # Assuming this function exists

        data = {'input': input, 'label': label}

        # Apply transform (if defined)
        if self.transform:
            data = self.transform(data)

        return data

## 세 가지의 transform 구현하기 
class ToTensor(object): # ToTensor(): numpy -> tensor
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Image의 numpy 차원 = (Y, X, CH)
        # Image의 tensor 차원 = (CH, Y, X)
        # CH 위치 옮기기
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std # label은 0 또는 1이라서 안함

        data = {'label': label, 'input': input}

        return data
    
class RandomFlip(object): # RandomFlip()
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5: # 50%의 확률로 
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data
