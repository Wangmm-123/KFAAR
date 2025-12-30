import torch
import torch.nn as nn

class Idencoder(nn.Module):
    def __init__(self):
        super(Idencoder, self).__init__()

        # 定义图像处理部分
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128 , 512)
        self.relu5 = nn.ReLU(inplace=True)
        # 定义密钥处理部分
        self.key_fc1 = nn.Linear(512+8, 512+8)
        self.key_fc2 = nn.Linear(512+8, 512)
        self.key_relu = nn.ReLU()

        # 定义特征输出部分


    def forward(self, x, key=None):
        # 图像处理部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)



        # 特征输出部分
        x = self.fc1(x)
        x = self.relu5(x)
        if key is not None:
            # 密钥处理部分
            x = torch.cat((x, key), dim=1)
            x = self.key_fc1(x)
            x = self.key_relu(x)
            x = self.key_fc2(x)
            # 将图像特征和密钥特征拼接起来

        return x
