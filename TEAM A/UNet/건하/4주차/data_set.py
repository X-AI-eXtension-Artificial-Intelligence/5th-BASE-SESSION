import os
import random
import shutil

# 데이터가 있는 폴더 경로 설정
data_folder = 'cityscapes_data'

# train과 val 폴더의 경로 설정
train_folder = os.path.join(data_folder, 'train')
val_folder = os.path.join(data_folder, 'val')

# train과 val 폴더 내 파일 목록 가져오기
train_files = os.listdir(train_folder)
val_files = os.listdir(val_folder)

# train과 val 폴더에서 각각 랜덤하게 500개 파일 선택
random_train_files = random.sample(train_files, min(200, len(train_files)))
random_val_files = random.sample(val_files, min(30, len(val_files)))

# 선택되지 않은 파일 삭제
for file in train_files:
    if file not in random_train_files:
        file_path = os.path.join(train_folder, file)
        os.remove(file_path)

for file in val_files:
    if file not in random_val_files:
        file_path = os.path.join(val_folder, file)
        os.remove(file_path)

print("train과 val 폴더에서 각각 랜덤하게 500개 파일만을 남겼습니다.")
