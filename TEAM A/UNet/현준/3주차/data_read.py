import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load dataset
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

# train, validation, test set generation
nframe_train =24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data,'train')
dir_save_val = os.path.join(dir_data,'val')
dir_save_test = os.path.join(dir_data,'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# frame random index
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# save train data
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_train, f'label{i}.npy'), label_)
    np.save(os.path.join(dir_save_train, f'input{i}.npy'), input_)

# save validation data
offset_nframe += nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_val, f'label{i}.npy'), label_)
    np.save(os.path.join(dir_save_val, f'input{i}.npy'), input_)

# save test data
offset_nframe += nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_label)

    np.save(os.path.join(dir_save_test, f'label{i}.npy'), label_)
    np.save(os.path.join(dir_save_test, f'input{i}.npy'), input_)

# data visualization
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show