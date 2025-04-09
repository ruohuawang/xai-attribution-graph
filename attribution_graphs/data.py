from datasets import load_dataset
import os

# 创建数据目录（如果不存在）
data_dir = os.path.join("c:\\codes\\XAI\\attribution_graphs", "data")
os.makedirs(data_dir, exist_ok=True)

# 指定下载到本地的 data 目录
ds = load_dataset("stas/openwebtext-10k", cache_dir=data_dir)

# 打印数据集信息
print(f"数据集已下载到: {data_dir}")
print(f"数据集信息: {ds}")