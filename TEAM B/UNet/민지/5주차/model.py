import torch
import torch.nn as nn

# Define Convolution Block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
        nn.ReLU(inplace=True)
    )

# Define cropping function
def crop(conv1, conv2):
    diff_size = conv1.size()[2] - conv2.size()[2]
    start = diff_size // 2
    end = conv1.size()[2] - start
    sub = end - start

    if sub % 2 != 0:
        end -= 1

    cropped_conv = conv1[:, :, start:end, start:end]
    return cropped_conv

# UNet Model Class
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.conv1_1 = conv_block(1, 64)
        self.conv1_2 = conv_block(64, 64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = conv_block(64, 128)
        self.conv2_2 = conv_block(128, 128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = conv_block(128, 256)
        self.conv3_2 = conv_block(256, 256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = conv_block(256, 512)
        self.conv4_2 = conv_block(512, 512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Center path
        self.conv5_1 = conv_block(512, 1024)
        self.conv5_2 = conv_block(1024, 1024)

        # Expansive path
        self.upconv4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.exp4_1 = conv_block(1024, 512)
        self.exp4_2 = conv_block(512, 512)

        self.upconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.exp3_1 = conv_block(512, 256)
        self.exp3_2 = conv_block(256, 256)

        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.exp2_1 = conv_block(256, 128)
        self.exp2_2 = conv_block(128, 128)

        self.upconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.exp1_1 = conv_block(128, 64)
        self.exp1_2 = conv_block(64, 64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)  # Adjust final output channels to 1

    # Connect layers
    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        upconv4 = self.upconv4(conv5_2)
        cropped_conv4_2 = crop(conv4_2, upconv4)
        cat4 = torch.cat((cropped_conv4_2, upconv4), dim=1)
        exp4_1 = self.exp4_1(cat4)
        exp4_2 = self.exp4_2(exp4_1)

        upconv3 = self.upconv3(exp4_2)
        cropped_conv3_2 = crop(conv3_2, upconv3)
        cat3 = torch.cat((cropped_conv3_2, upconv3), dim=1)
        exp3_1 = self.exp3_1(cat3)
        exp3_2 = self.exp3_2(exp3_1)

        upconv2 = self.upconv2(exp3_2)
        cropped_conv2_2 = crop(conv2_2, upconv2)
        cat2 = torch.cat((cropped_conv2_2, upconv2), dim=1)
        exp2_1 = self.exp2_1(cat2)
        exp2_2 = self.exp2_2(exp2_1)

        upconv1 = self.upconv1(exp2_2)
        cropped_conv1_2 = crop(conv1_2, upconv1)
        cat1 = torch.cat((cropped_conv1_2, upconv1), dim=1)
        exp1_1 = self.exp1_1(cat1)
        exp1_2 = self.exp1_2(exp1_1)

        fc = self.fc(exp1_2)

        return fc
