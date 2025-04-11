import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 模型配置
# MODEL_CONFIG = {
#     "hidden_size": 768,
#     "num_layers": 12,
#     "num_heads": 12,
#     "dropout": 0.1,
#     "vocab_size": 50257,  # GPT-2 词汇表大小
# }

# # 训练配置
# TRAIN_CONFIG = {
#     "batch_size": 4,
#     "learning_rate": 3e-5,
#     "weight_decay": 0.01,
#     "epochs": 10,
#     "warmup_steps": 1000,
#     "max_grad_norm": 1.0,
#     "gradient_accumulation_steps": 4,  # 梯度累积步数，相当于将有效批量大小扩大4倍
#     "effective_batch_size": 4 * 4,    # 有效批量大小 = batch_size * gradient_accumulation_steps
#     "fp16": True,                      # 是否使用混合精度训练
# }

# 归因图配置
ATTRIBUTION_CONFIG = {
    "num_samples": 100,  # 用于计算归因图的样本数
    "edge_threshold": 0.01,  # 归因图中边的阈值
}

# 主配置
CONFIG = {
    "model_path": "C:\\codes\\llms\\qwen05b",  # 更新为本地Qwen模型路径
    "data_path": "c:/codes/XAI/attribution_graphs/data/openwebtext-10k-train.arrow",
    "output_dir": "c:/codes/XAI/attribution_graphs/results",
    "clt_dir": "c:/codes/XAI/attribution_graphs/results/clt",
    "batch_size": 4,
    "seq_length": 128,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "attribution_samples": 10,
    "edge_threshold": 0.01
}

# CLT配置
CLT_CONFIG = {
    "hidden_size": 896,  
    "num_layers": 12,    
    "features_per_layer": 4096,  
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 1,
    "epochs": 10,
    "save_interval": 1,
    "max_grad_norm": 1.0,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gradient_accumulation_steps": 16,  
    "effective_batch_size": 1 * 16,     # 有效批量大小
    "fp16": True,                      # 是否使用混合精度训练
    "auto_update_config": False        
}