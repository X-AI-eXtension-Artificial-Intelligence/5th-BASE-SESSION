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
transform = transforms.Compose( 
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR100 datasets 불러오기
train_dataset = CIFAR100Dataset(root="./data/", train=True, transform=transform)
test_dataset = CIFAR100Dataset(root="./data/", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# VGG16 인스턴스 생성
model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_lst = []
for i in range(num_epoch): 
    for k,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_= label.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(f'{i} epoch loss : {loss}')
        loss_lst.append(loss.cpu().detach().numpy())


    # 학습 후 모델 저장
    torch.save(model.state_dict(), './VGG16_cifar100.pth')