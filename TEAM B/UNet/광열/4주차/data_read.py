import os
import numpy as np
from PIL import Image
from tqdm import tqdm

img_dir = './data2/imgs'  
mask_dir = './data2/masks'  
dir_data = './datasets2'  

nframe_train = 4888  
nframe_val = 100  
nframe_test = 100 

img_files = sorted(os.listdir(img_dir))
mask_files = sorted(os.listdir(mask_dir))

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

def save_data(data_dir, frame_range):
    for i in tqdm(frame_range):
        img_path = os.path.join(img_dir, img_files[i])
        mask_path = os.path.join(mask_dir, mask_files[i])

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        img_arr = np.array(img)
        mask_arr = np.array(mask)

        np.save(os.path.join(data_dir, f'input_{i:03d}.npy'), img_arr)
        np.save(os.path.join(data_dir, f'label_{i:03d}.npy'), mask_arr)

save_data(dir_save_train, range(nframe_train))

save_data(dir_save_val, range(nframe_train, nframe_train + nframe_val))

save_data(dir_save_test, range(nframe_train + nframe_val, nframe_train + nframe_val + nframe_test))
