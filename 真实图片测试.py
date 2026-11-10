import json
import os

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from 数据集类 import BASE_DIR, IMAGE_SIZE
from ResNet18模型 import build_module

# ==================== 图片绝对路径 ====================
IMAGE_PATH = r"/Users/davidzero/CLionProjects/水果分类/汽车识别/torchvision-stanford-cars/versions/2/stanford_cars/cars_test/00001.jpg"

# 获取设备，与训练时保持一致
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# 加载图片
def loadImage(path):
    image = Image.open(path).convert("RGB")
    transform = Compose(
        [
            Resize(IMAGE_SIZE),
            ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


# 加载模型
def loadModule(path):
    Module = build_module(pretrained=False)
    Module.load_state_dict(torch.load(path, map_location=device))
    Module.to(device)
    Module.eval()  # 测评模式，参数不再调整
    return Module


def tool(output, labels, topk=5):
    """把模型输出转换成 top-k 的 (类别名, 概率)"""
    probs = torch.softmax(output.detach(), dim=1).squeeze(0)
    values, indices = torch.topk(probs, topk)
    return [(labels[str(int(index))], float(value)) for value, index in zip(values, indices)]


if __name__ == "__main__":
    # 直接使用文件顶部 IMAGE_PATH 中填写的图片路径
    image_path = IMAGE_PATH
    if not os.path.exists(image_path):
        print(f"图片不存在：{image_path}")
        print("请修改 真实图片测试.py 顶部的 IMAGE_PATH，填写正确的图片绝对路径。")
        exit(1)

    image = loadImage(image_path)
    Module = loadModule(os.path.join(BASE_DIR, "Module.pt"))
    output = Module(image)

    with open(os.path.join(BASE_DIR, "label.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)

    print(f"预测图片：{image_path}")
    # 只取概率最大的那个结果
    name, prob = tool(output, labels, topk=1)[0]
    print(f"预测结果：{name}  概率：{prob * 100:.2f}%")
