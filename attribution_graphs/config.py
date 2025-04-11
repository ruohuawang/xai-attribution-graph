import torch
import os

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 归因图配置
ATTRIBUTION_CONFIG = {
    "num_samples": 100,  # 用于计算归因图的样本数
    "edge_threshold": 0.01,  # 归因图中边的阈值
}

# 模型路径配置
MODEL_PATHS = {
    "qwen2": "C:\\codes\\XAI\\attribution_graphs\\llms\\qwen05b",
    "gpt2": "C:\\codes\\XAI\\attribution_graphs\\llms\\gpt2",  # 使用Hugging Face模型名称
    "bert": "C:\\codes\\XAI\\attribution_graphs\\llms\\bert"  # 使用Hugging Face模型名称
}

# 主配置
CONFIG = {
    "model_type": "gpt2",  # 模型类型，可选值为 "qwen2", "gpt2", "bert"
    "data_path": "c:/codes/XAI/attribution_graphs/data/openwebtext-10k-train.arrow",
    "output_dir": "c:/codes/XAI/attribution_graphs/results",
    "clt_dir": "c:/codes/XAI/attribution_graphs/results/clt",
    "batch_size": 4,
    "seq_length": 1024,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "attribution_samples": 10,
    "edge_threshold": 0.01
}

CONFIG["model_path"] = MODEL_PATHS[CONFIG["model_type"]]

# CLT配置
CLT_CONFIG = {
    "hidden_size": 768,  #768,12 for bert&gpt2, 896,24 for qwen2
    "num_layers": 12,    # <= num of layers in model
    "features_per_layer": 3000,  
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 1,
    "epochs": 10,
    "save_interval": 1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gradient_accumulation_steps": 8,  # 梯度累积步数
    "effective_batch_size": 1 * 8,     # 有效批量大小 = batch_size * gradient_accumulation_steps
    "fp16": True,                      # 是否使用混合精度训练
    "auto_update_config": False        # 禁止自动更新
}