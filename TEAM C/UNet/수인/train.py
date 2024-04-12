## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms
import torchvision.transforms.functional as F

## Parser 생성하기
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--result_dir", default="result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './datasets' # 데이터가 저장되어 있는 디렉토리
ckpt_dir = './checkpoint' # 학습된 네트워크가 저장될 디렉토리
log_dir = './log' # 텐서보드 로그 파일이 저장될 디렉토리

result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 디렉토리 생성하기
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    # batchsize에 의해 나누어지는 데이터셋의 수 계산 
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
# 아웃풋을 저장하기 위한 몇 가지 함수 정의  
# tonumppy: from tensor to numpy 
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# normalization된 걸 다시 denormalization 시키기 
fn_denorm = lambda x, mean, std: (x * std) + mean
# 네트워크 아웃풋의 이미지를 binary 클래스로 분류
fn_class = lambda x: 1.0 * (x > 0.5)

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0 # training이 시작되는 epoch의 포지션을 0으로 설정

# input : 960 x 960 / output 572 x 572 4개
def divide_input(img):
    left_top = F.crop(img, 0, 0, 572, 572)
    right_top = F.crop(img, 388, 0, 572, 572)
    left_bottom = F.crop(img, 0, 388, 572, 572)
    right_bottom = F.crop(img, 388, 388, 572, 572)
    return left_top, right_top, left_bottom, right_bottom

# input : 960 x 960 / output 388 x 388 4개
def divide_label(img):
    left_top = F.crop(img, 92, 92, 388, 388)
    right_top = F.crop(img, 480, 92, 388, 388)
    left_bottom = F.crop(img, 92, 480, 388, 388)
    right_bottom = F.crop(img, 480, 480, 388, 388)
    return left_top, right_top, left_bottom, right_bottom

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        # 저장된 네트워크를 불러와서 연속적으로 학습시킬 수 있도록 네트워크 로드하기 
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # 네트워크에게 train모드임을 알려주기 위해 train func을 activate함
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward
            label = data['label'].to(device) 
            inputs = data['image'].to(device)

            input_0, input_1, input_2, input_3 = divide_input(inputs)
            label_0, label_1, label_2, label_3 = divide_label(label)

            input_list = []
            label_list = []

            input_list.append(input_0)
            input_list.append(input_1)
            input_list.append(input_2)
            input_list.append(input_3)

            label_list.append(label_0)
            label_list.append(label_1)
            label_list.append(label_2)
            label_list.append(label_3)

            for i in range(4):
                print('input shape : ', input_list[i].shape)
                print('label shape : ', label_list[i].shape)
                output = net(input_list[i])

                print('output shape : ', output.shape)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # label, input, output을 Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

         # loss를 Tesorboard에 저장하기
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # 네트워크를 validation하는 부분 
        with torch.no_grad(): # backward을 막기 위해 no_grad()
            net.eval() # validation 모드 
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard 저장하기
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # 에폭이 50번마다 진행될때마다 저장
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    # 학습 완료되면 close 
    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # Tensorboard 저장하기
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_arr)))
