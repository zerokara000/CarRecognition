import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor

from 数据集类 import BASE_DIR, IMAGE_SIZE, MyDataSet
from 神经网络 import MyNN

# 获取设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 数据增强：训练集随机水平翻转，测试集只做缩放
train_transform = Compose(
    [
        Resize(IMAGE_SIZE),
        RandomHorizontalFlip(),
        ToTensor(),
    ]
)
test_transform = Compose(
    [
        Resize(IMAGE_SIZE),
        ToTensor(),
    ]
)

# 数据
train_set = MyDataSet(os.path.join(BASE_DIR, "trainset_lable.json"), transform=train_transform)
test_set = MyDataSet(os.path.join(BASE_DIR, "testset_lable.json"), transform=test_transform)

# 超参
batch_size = 64
lr = 0.001
epoch = 10

# 加载数据
train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# 创建模型（从零开始训练）
Module = MyNN().to(device)
# 创建优化器
op = torch.optim.Adam(Module.parameters(), lr)
# 创建损失函数
loss_f = torch.nn.CrossEntropyLoss()


def train_one_epoch(module, loader, optimizer):
    """训练一个 epoch，返回平均损失和准确率"""
    module.train()
    total_loss = 0.0
    true = 0
    count = 0
    for j, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 向前传播
        outputs = module(images)
        # 计算损失
        loss = loss_f(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        true += (torch.argmax(outputs, dim=1) == labels).sum().item()
        count += images.size(0)
        if (j + 1) % 20 == 0:
            print(f"  第{j + 1}/{len(loader)}次，当前损失:{loss.item():.4f}")

    return total_loss / count, true / count


def evalModule(module, loader):
    """在测试集上评估，返回平均损失和准确率"""
    module.eval()  # 开启测评模式：参数不再调整，使用更稳定的计算方式
    total_loss = 0.0
    true = 0
    count = 0
    with torch.no_grad():  # 测评不需要计算梯度
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = module(images)
            loss = loss_f(outputs, labels)

            total_loss += loss.item() * images.size(0)
            true += (torch.argmax(outputs, dim=1) == labels).sum().item()
            count += images.size(0)

    print(f"测试结束，当前测试{count}次，正确{true}次，正确率{true / count * 100:.2f}%")
    return total_loss / count, true / count


if __name__ == "__main__":
    print(f"训练设备：{device}，训练集：{len(train_set)}，测试集：{len(test_set)}")

    best_acc = 0.0
    for i in range(epoch):
        start = time.time()
        train_loss, train_acc = train_one_epoch(Module, train_data, op)
        test_loss, test_acc = evalModule(Module, test_data)
        print(
            f"第{i + 1}/{epoch}纪元，"
            f"训练损失:{train_loss:.4f}，训练准确率:{train_acc * 100:.2f}%，"
            f"测试损失:{test_loss:.4f}，耗时:{time.time() - start:.0f}秒"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存当前最好的模型参数
            # 保存 MyNN 的权重，避免覆盖测试用的 ResNet18 模型 Module.pt
            torch.save(Module.state_dict(), os.path.join(BASE_DIR, "Module_MyNN.pt"))
            print(f"  已保存更好的模型，测试准确率:{test_acc * 100:.2f}%")

    print(f"训练完成，最佳测试准确率：{best_acc * 100:.2f}%")
    print(f"模型已保存到：{os.path.join(BASE_DIR, 'Module_MyNN.pt')}")
