import json
import os

import scipy.io as sio

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "torchvision-stanford-cars", "versions", "2", "stanford_cars")
TEST_DIR = os.path.join(DATA_DIR, "cars_test")
TEST_ANNO = os.path.join(DATA_DIR, "cars_test_annos_withlabels.mat")
META = os.path.join(DATA_DIR, "devkit", "cars_meta.mat")


def load_class_names():
    """从 cars_meta.mat 读取 196 个类别名称，返回 0 起始的列表"""
    meta = sio.loadmat(META)
    return [str(name[0]) for name in meta["class_names"][0]]


def main():
    class_names = load_class_names()
    # 测试集标注同样带有 class 字段（cars_test_annos_withlabels.mat）
    annotations = sio.loadmat(TEST_ANNO)["annotations"][0]
    labels = {}
    for index, anno in enumerate(annotations):
        class_id = int(anno["class"][0][0]) - 1
        image_path = os.path.join(TEST_DIR, str(anno["fname"][0]))
        labels[str(index)] = {
            "image": image_path,
            "labels": class_names[class_id],
        }

    with open(os.path.join(BASE_DIR, "testset_lable.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f"测试集图片数量：{len(labels)}，已保存 testset_lable.json")


if __name__ == "__main__":
    main()
