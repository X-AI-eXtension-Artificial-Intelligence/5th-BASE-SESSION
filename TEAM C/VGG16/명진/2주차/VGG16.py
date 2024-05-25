import torch.nn as nn
#import torch.nn.functional as F 

# 아키텍쳐 구현 전에 conv layer 2개, 3개짜리 시퀀스 생성해서 사용
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

# conv layer 3개
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

class VGG16(nn.Module):
    def __init__(self, base_dim, num_classes=1000):
        super(VGG16, self).__init__()
        # conv 레이어 파트
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 3 -> 64 (입력채널 -> 출력채널)
            conv_2_block(base_dim, 2 * base_dim), # 64 -> 128
            conv_3_block(2 * base_dim, 4 * base_dim), # 128 -> 256
            conv_3_block(4 * base_dim, 8 * base_dim), # 256 -> 512
            conv_3_block(8 * base_dim, 8 * base_dim), # 512 -> 512       
        )
        # FC 레이어 파트
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8 * base_dim * 7 * 7, 4096), # 224x224 이미지를 입력으로 받았을 때의 크기 계산
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes) # 출력 레이어
        )
        
    # Forward pass 정의, 각 레이어 순차 통과하도록 함
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1) # 
        x = self.fc_layer(x)
        return x
