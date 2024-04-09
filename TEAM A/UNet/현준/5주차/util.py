import os
import numpy as np

import torch
import torch.nn as nn

## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

def center_crop(output, crop_size):
    _, _, h, w = output.shape
    crop_h, crop_w = crop_size[0], crop_size[1]
    
    # 시작 인덱스를 계산
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    # 가운데 영역을 크롭
    cropped_output = output[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    return cropped_output