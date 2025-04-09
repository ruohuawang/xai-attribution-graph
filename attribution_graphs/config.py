import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置
MODEL_CONFIG = {
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "dropout": 0.1,
    "vocab_size": 50257,  # GPT-2 词汇表大小
}

# 训练配置
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "epochs": 10,
    "warmup_steps": 1000,
    "max_grad_norm": 1.0,
}

# 归因图配置
ATTRIBUTION_CONFIG = {
    "num_samples": 100,  # 用于计算归因图的样本数
    "edge_threshold": 0.01,  # 归因图中边的阈值
}