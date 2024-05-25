import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from VGG16 import VGG16

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cuda gpu사용

# hyperparameter
batch_size = 100

normalize = transforms.Normalize( # 정규화
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225],   # ImageNet std
) # 색상 rgb normalize

# define transforms: 데이터 전처리
transform = transforms.Compose([
    transforms.Resize(256),  # 이미지 크기 조정
    transforms.CenterCrop(224),  # 중앙 부분 크롭
    transforms.ToTensor(),
    normalize, # 정규화
])

# ImageNet 데이터셋 load
imagenet_test = datasets.ImageFolder(root='/path/to/imagenet/val', transform=transform)

test_loader = DataLoader(imagenet_test, batch_size=batch_size)

model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model/VGG16_01.pth'))

# Test
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, predicted = torch.max(output, 1) # 여기서 1은 뭐지? #인덱스 찾는거임

        total += labels.size(0)
        correct += (predicted == labels).sum().float()

    # 테스트 데이터셋에 따른 네트워크 정확도(맞춘거/전체)
    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
