import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from vgg16 import VGG16

batch_size = 64
learning_rate = 0.0001
num_epoch = 30

# device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset Class 생성
class CIFAR100Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.cifar100 = datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        image, label = self.cifar100[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# transform > 이미지 텐서로 변환 및 정규화
transform = transforms.Compose( # (데이터 정규화)
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR100 datasets 불러오기
test_dataset = CIFAR100Dataset(root="./data/", train=False, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size)


# VGG16 인스턴스 생성
model = VGG16(base_dim=64).to(device)
# pre trained model load
model.load_state_dict(torch.load('./VGG16_cifar100.pth'))

correct = 0 # correct count
total = 0 # total count

model.eval()

# 가중치 갱신 X
with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output,1) 

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))
    