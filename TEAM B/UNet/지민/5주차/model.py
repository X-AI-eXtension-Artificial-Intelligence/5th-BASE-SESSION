## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        ## 구조 사진 보면서 진행 - 파란색 화살표 conv -> batch normalization -> ReLU (미리 함수에서 정의)
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]   # conv layer
            layers += [nn.BatchNorm2d(num_features=out_channels)]   # batch normalization layer
            layers += [nn.ReLU()]   # ReLU layer

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # 왼쪽 enc, 오른쪽 dec
        # 첫번째 스테이지의 파란색 화살표들
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # kernel_size, stride, padding, bias는 고정
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # 빨간색 화살표 - maxpooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)  # encoder part 완료

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)  # encoder의 5번째 stage와 반대

        # 초록색 화살표 - upconv 2x2
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)   # 구조 보면 아래 스테이지에서 올라온 512와 enc에서 온 512 합쳐짐 -> 2배
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)   # enc4_1과 대응

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)  # skopconnection으로 연결되는 volume 존재하기 때문에 2배
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # 녹색 화살표 - 1x1 conv
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # U-Net layer 연결하기
    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=1: 채널 방향 / 0: batch, 2: height, 3: width
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


# ## 데이터 로더를 구현하기
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform

#         # 데이터 디렉토리에 있는 모든 파일들의 리스트
#         lst_data = os.listdir(self.data_dir)

#         # label과 input 데이터 나눔
#         lst_label = [f for f in lst_data if f.startswith('label')]
#         lst_input = [f for f in lst_data if f.startswith('input')]

#         lst_label.sort()
#         lst_input.sort()

#         # 이 클래스의 파라미터로 지정
#         self.lst_label = lst_label
#         self.lst_input = lst_input

#     def __len__(self):
#         return len(self.lst_label)

#     # 실제로 데이터를 get하는 함수
#     def __getitem__(self, index):
#         label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
#         input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

#         # 데이터 normalize
#         label = label/255.0
#         input = input/255.0

#         # 무슨 말이됴?
#         if label.ndim == 2:
#             label = label[:, :, np.newaxis]  # 라벨의 마지막 axis를 임의로 생성
#         if input.ndim == 2:
#             input = input[:, :, np.newaxis]

#         data = {'input': input, 'label': label}  # 딕셔너리 형태로

#         if self.transform:
#             data = self.transform(data)

#         return data

# ## 중간점검 코드
# import matplotlib.pyplot as plt
# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
# data = dataset_train.__getitem__(0)

# input = data['input']
# label = data['label']

# plt.subplot(121)
# plt.imshow(input.squeeze())  # 에러 발생 -> squeeze 써서 1인 dimension 갖는 axis 제거

# plt.subplot(122)
# plt.imshow(label.squeeze())

# plt.show()


# ## 트랜스폼 구현하기
# class ToTensor(object):  # numpy(Y, X, CH) -> tensor(CH, Y, X)
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         label = label.transpose((2, 0, 1)).astype(np.float32)  # 넘파이의 채널을 첫번째로 옮기고 나머지는 그대로
#         input = input.transpose((2, 0, 1)).astype(np.float32)

#         # 다시 딕셔너리로
#         # 넘파이를 텐서로: from_numpy
#         data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

#         return data

# class Normalization(object):
#     def __init__(self, mean=0.5, std=0.5):
#         self.mean = mean
#         self.std = std

#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         input = (input - self.mean) / self.std  # label은 0 또는 1이기 때문에 적용 X

#         data = {'label': label, 'input': input}

#         return data

# class RandomFlip(object):
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         if np.random.rand() > 0.5:
#             label = np.fliplr(label)  # input과 lable은 항상 동시에 flip
#             input = np.fliplr(input)

#         if np.random.rand() > 0.5:
#             label = np.flipud(label)
#             input = np.flipud(input)

#         data = {'label': label, 'input': input}

#         return data

# ## test 코드
# import matplotlib.pyplot as plt

# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform = transform)

# data = dataset_train.__getitem__(0)

# input = data['input']
# label = data['label']

# plt.subplot(121)
# plt.imshow(input.squeeze())  # 에러 발생 -> squeeze 써서 1인 dimension 갖는 axis 제거

# plt.subplot(122)
# plt.imshow(label.squeeze())

# plt.show()
# 결과 보면 어두운 부분은 -1쪽으로 normalization, 노란색 밝은 부분은 1에 가깝게 normalization -> (-1,1)

# ## 네트워크 학습하기
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
# loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

# dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
# loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

# ## 네트워크 생성하기
# net = UNet().to(device)

# ## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)

# ## Optimizer 설정하기 - Adam
# optim = torch.optim.Adam(net.parameters(), lr=lr)

# # 그밖에 부수적인 variables 설정하기
# num_data_train = len(dataset_train)
# num_data_val = len(dataset_val)

# num_batch_train = np.ceil(num_data_train / batch_size)
# num_batch_val = np.ceil(num_data_val / batch_size)

# ## 그밖에 부수적인 functions 설정하기
# fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # tensor -> numpy
# fn_denorm = lambda x, mean, std: (x * std) + mean  # denormalization
# fn_class = lambda x: 1.0 * (x > 0.5)   # 네트워크 아웃풋 이미지를 binary로

# ## Tensorboard 를 사용하기 위한 SummaryWriter 설정
# writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

# ## 네트워크 저장하기
# def save(ckpt_dir, net, optim, epoch):
#     if not os.path.exists(ckpt_dir)
#         os.makedirs(ckpt_dir)
    
#     torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
#                 "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

# ## 네트워크 불러오기
# def load(ckpt_dir, net, optim):
#     if not os.path.exists(ckpt_dir):
#         epoch = 0
#         return net, optim, epoch

#     ckpt_lst = os.listdir(ckpt_dir)
#     ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

#     dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

#     net.load_state_dict(dict_model['net'])
#     optim.load_state_dict(dict_model['optim'])
#     epoch = int(ckpt_lst[-1].split('epoch')[-1].split('pth')[0])

#     return net, optim, epoch

# ## 네트워크 학습시키기
# st_epoch = 0
# ## 저장이 되어있는 네트워크가 있다면 네트워크를 불러와서 네트워크를 연속해서 학습시킬 수 있도록
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# for epoch in range(st_epoch + 1, num_epoch + 1):
#     net.train()  # train모드임을 알려준다
#     loss_arr = []

#     for batch, data in enumerate(loader_train, 1):
#         # forward pass
#         label = data['label'].to(device)
#         input = data['input'].to(device)

#         output = net(input)

#         # backward pass
#         optim.zero_grad()

#         loss = fn_loss(output, label)
#         loss.backward()

#         optim.step()

#         # 손실함수 계산
#         loss_arr += [loss.item()]

#         print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
#                 (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

#         # Tensorboard 저장하기
#         label = fn_tonumpy(label)
#         input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
#         output = fn_tonumpy(fn_class(output))

#         writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
#         writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
#         writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

#     writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

#     # validation -back propagation X 
#     with torch.no_grad():  # back propagation 사전 방지
#         net.eval()  # 현재 val모드임을 알린다
#         loss_arr = []

#         for batch, data in enumerate(loader_val, 1):
#             # forward pass
#             label = data['label'].to(device)
#             input = data['input'].to(device)

#             output = net(input)

#             # 손실함수 계산하기
#             loss = fn_loss(output, label)

#             loss_arr += [loss.item()]

#             print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
#                     (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

#             # Tensorboard 저장하기
#             label = fn_tonumpy(label)
#             input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
#             output = fn_tonumpy(fn_class(output))

#             writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
#             writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
#             writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

#     writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

#     # 에폭마다 네트워크를 저장
#     if epoch % 50 == 0:
#         save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

# writer_train.close()
# writer_val.close()
