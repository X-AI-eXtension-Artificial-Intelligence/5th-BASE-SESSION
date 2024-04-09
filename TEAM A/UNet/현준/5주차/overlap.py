import numpy as np
import matplotlib.pyplot as plt

def overlap_tile(image, pad_width):
    # 상하 부분을 반전하여 추출하고 원본 이미지에 붙임
    top = image[:pad_width, :][::-1, :]
    bottom = image[-pad_width:, :][::-1, :]
    padded_tb = np.concatenate([top, image, bottom], axis=0)

    # 좌우 부분을 반전하여 추출
    # 좌우 패딩을 추가하기 전에, padded_tb의 높이에 맞게 left와 right 패딩의 크기를 조정
    left = padded_tb[:, :pad_width][:, ::-1]
    right = padded_tb[:, -pad_width:][:, ::-1]

    # 좌우 패딩을 추가
    padded_complete = np.concatenate([left, padded_tb, right], axis=1)

    return padded_complete

# .npy 파일에서 이미지 데이터 로드
image_path = 'datasets/train/input0.npy'  # .npy 이미지 파일 경로
image = np.load(image_path)

# overlap_tile 함수를 사용하여 이미지에 패딩 적용
pad_width = 30  # 패딩 크기 설정
padded_image = overlap_tile(image, pad_width)

# 결과 이미지 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')  # 원본 이미지 표시
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Padded Image')
plt.imshow(padded_image, cmap='gray')  # 패딩 적용된 이미지 표시
plt.axis('off')

plt.show()