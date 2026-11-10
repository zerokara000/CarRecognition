# 汽车识别

基于 PyTorch 的斯坦福汽车数据集 (Stanford Cars) 图像分类项目。

## 项目结构

| 文件 | 说明 |
| --- | --- |
| `ResNet18模型.py` | 基于 torchvision 预训练 ResNet18 微调的分类模型 |
| `神经网络.py` | 自定义 CNN (MyNN) 网络结构 |
| `模型训练.py` | 训练入口, 保存权重到 `Module.pt` |
| `数据集类.py` | PyTorch Dataset 封装, 读取标签 json 并加载图片 |
| `数据集下载.py` | 用 kagglehub 下载数据集 |
| `标签文件生成.py` / `测试标签文件生成.py` | 从 scipy.io 的标注文件生成训练/测试标签 json |
| `标签对应文件.py` / `标签文件生成.py` | 类别名与编号的映射 |
| `真实图片测试.py` | 加载训练好的模型对单张真实图片做推理 |

## 使用方法

```bash
pip install -r requirements.txt
python 数据集下载.py        # 下载数据集到 torchvision-stanford-cars/
python 标签文件生成.py       # 生成 trainset_lable.json
python 测试标签文件生成.py    # 生成 testset_lable.json
python 模型训练.py           # 训练, 输出 Module.pt
python 真实图片测试.py        # 用训练好的模型测试真实图片
```

## 说明

- 数据集目录 `torchvision-stanford-cars/` (约 1.9GB) 不纳入版本控制。
- 较大的模型权重 `Module.pt`、`Module_ResNet18_backup.pt` 不入库;
  小的自定义网络权重 `Module_MyNN.pt` 保留在仓库中。
