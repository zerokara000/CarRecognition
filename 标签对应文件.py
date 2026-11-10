import json
import os

# 项目根目录（本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "label.json"), "r", encoding="utf-8") as f:
    label = json.load(f)

# 生成 类别名 -> 编号 的映射
str_to_number = {name.lower(): int(index) for index, name in label.items()}

with open(os.path.join(BASE_DIR, "str_to_number.json"), "w", encoding="utf-8") as f:
    json.dump(str_to_number, f, ensure_ascii=False, indent=2)

print(f"类别数量：{len(str_to_number)}，已保存 str_to_number.json")
