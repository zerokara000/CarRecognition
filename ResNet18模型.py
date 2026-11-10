import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# Stanford Cars 一共 196 个类别
NUM_CLASSES = 196


def build_module(pretrained=False, freeze_backbone=False):
    """
    创建 ResNet18 模型。
    pretrained：是否加载 ImageNet 预训练权重（第一次运行会自动下载）
    freeze_backbone：是否冻结卷积层，只训练最后的分类层
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    module = resnet18(weights=weights)

    # 原来的 ResNet18 输出 1000 类，这里换成 196 类
    module.fc = nn.Linear(module.fc.in_features, NUM_CLASSES)

    if freeze_backbone:
        # 冻结除分类层以外的所有参数
        for name, param in module.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return module
