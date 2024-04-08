import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose
## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

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
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0
        input = input/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

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

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

def mirroring_Extrapolate(img_tensor):
    # img_tensor: PyTorch tensor of shape (C, H, W)

    C, H, W = img_tensor.shape

    # 이미지의 높이와 너비에 따라 패딩 양 계산

    if H < 388:
        pad_H_top = (572 - H) // 2
        pad_H_bottom = (572 - H) - pad_H_top
    else:
        pad_H_top = 92
        pad_H_bottom = 388 - (H % 388) + 92

    if W < 388:
        pad_W_left = (572 - W) // 2
        pad_W_right = (572 - W) - pad_W_left
    else:
        pad_W_left = 92
        pad_W_right = 388 - (W % 388) + 92

    # Reflect padding
    # 미러링 패딩을 사용하여 이미지의 부족한 부분을 채우기
    img_padded = torch.nn.functional.pad(img_tensor, (pad_W_left, pad_W_right, pad_H_top, pad_H_bottom), mode='reflect')

    return img_padded

#  이미지의 가장자리를 복사하여 반대편에 붙여 넣어서 패딩을 채우기 
class CustomMirroringExtrapolate(object):
    def __call__(self, sample):
        input, label = sample['input'], sample['label']

        # Assuming input and label are already tensors from a previous transform
        input_padded = mirroring_Extrapolate(input)
        label_padded = mirroring_Extrapolate(label)

        return {'input': input_padded, 'label': label_padded}

transform = Compose([
    ToTensor(),
    CustomMirroringExtrapolate(),
    Normalization(mean=0.5, std=0.5),
    RandomFlip(),
])
