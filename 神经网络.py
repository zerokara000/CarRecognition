import torch.nn as nn

# 输入图片尺寸
IMAGE_SIZE = (224, 224)
# Stanford Cars 一共 196 个类别
NUM_CLASSES = 196


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.Net = nn.Sequential(
            # 第一次卷积：3 通道输入
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 第二次卷积：224*224-->112*112
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 56*56
            # 第三次卷积
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 28*28
            # 第四次卷积
            nn.Conv2d(128, 256, 3),
            # 26*26
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 13*13
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 输出 196 个车型类别
            nn.Linear(128, NUM_CLASSES),
        )

    # 向前传播
    def forward(self, X):
        return self.Net(X)
