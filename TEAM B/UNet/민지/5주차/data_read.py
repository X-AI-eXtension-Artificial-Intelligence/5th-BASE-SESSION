#첫번째 파일 
## 필요한 패키지 등록
#%%

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = '/home/work/XAI/XAI/week3/code/datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

## 데이터셋 나누어서 저장하기 - 30프레임 중 분배하기
nframe_train = 24
nframe_val = 3
nframe_test = 3

#디렉토리 저장 
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

#디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train) #train 데이터셋 

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val) #val 데이터셋

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test) #test 데이터셋

## 데이터셋을 랜덤하게 저장하기 위해 랜덤 인덱스 생성 
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## train 데이터셋 저장
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## val 데이터셋 저장 
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## test 데이터셋 저장 
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

##

plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
#그림은 안 보이지만, segmentation된 이미지가 나온다.