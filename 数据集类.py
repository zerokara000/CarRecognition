import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#输入图片尺寸
IMAGE_SIZE = (224, 224)


class MyDataSet(Dataset):
    # transform：将图片转化为张量的工具
    # target_transform：将字符串转化为标签的工具
    def __init__(self, file, transform=None, target_transform=None):
        # 加载标签文件（json 中保存了图片路径和类别名）
        with open(file, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        if transform is None:
            self.transform = self.build_transform()
        else:
            self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(BASE_DIR, "str_to_number.json"), "r", encoding="utf-8") as f:
            self.str_number = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        item = str(item)  # 转换为字符串类型
        path = self.labels[item]["image"]  # 获取图片地址
        label = self.labels[item]["labels"]  # 获取图片标签
        # 统一转换为 RGB，避免灰度图导致通道数不匹配
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        # 统一转成小写，与 str_to_number.json 的键保持一致
        label = label.lower()
        if self.target_transform is None:
            label = self.build_target_transform(label)
        else:
            label = self.target_transform(label)
        return image, label

    def build_target_transform(self, s):
        return self.str_number[s]

    def build_transform(self):
        transform = Compose(
            [
                Resize(IMAGE_SIZE),
                ToTensor(),
            ]
        )
        return transform


if __name__ == "__main__":
    train_set = MyDataSet(os.path.join(BASE_DIR, "trainset_lable.json"))
    image, label = train_set[20]
    print(f"数据集大小：{len(train_set)}")
    print(f"图片形状：{tuple(image.shape)}，标签编号：{label}")
