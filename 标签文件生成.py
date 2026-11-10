import json
import os

import scipy.io as sio

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "torchvision-stanford-cars", "versions", "2", "stanford_cars")
TRAIN_DIR = os.path.join(DATA_DIR, "cars_train")
TRAIN_ANNO = os.path.join(DATA_DIR, "devkit", "cars_train_annos.mat")
META = os.path.join(DATA_DIR, "devkit", "cars_meta.mat")


def load_class_names():
    """从 cars_meta.mat 读取 196 个类别名称，返回 0 起始的列表"""
    meta = sio.loadmat(META)
    # class_names 是 1x196 的 cell 数组
    names = [str(name[0]) for name in meta["class_names"][0]]
    return names


def main():
    class_names = load_class_names()
    # 保存 编号 -> 类别名 的映射，供真实图片测试使用
    label = {str(i): name for i, name in enumerate(class_names)}
    with open(os.path.join(BASE_DIR, "label.json"), "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    print(f"类别数量：{len(class_names)}，已保存 label.json")

    # 读取训练集标注
    annotations = sio.loadmat(TRAIN_ANNO)["annotations"][0]
    labels = {}
    for index, anno in enumerate(annotations):
        # class 从 1 开始，转换为 0 起始
        class_id = int(anno["class"][0][0]) - 1
        # fname 是图片文件名，拼成完整路径
        image_path = os.path.join(TRAIN_DIR, str(anno["fname"][0]))
        labels[str(index)] = {
            "image": image_path,
            "labels": class_names[class_id],
        }

    with open(os.path.join(BASE_DIR, "trainset_lable.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"训练集图片数量：{len(labels)}，已保存 trainset_lable.json")


if __name__ == "__main__":
    main()
