import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import random

from data_loader import get_dataloader
from model_loader import load_model
from models.clt import CrossLayerTranscoder, ReplacementModel, LocalReplacementModel
from attribution_clt import AttributionGraphCLT

# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 配置
CONFIG = {
    "model_path": "C:\\codes\\llms\\qwen05b",
    "data_path": "C:\\codes\\XAI\\attribution_graphs\\data\\stas___openwebtext-10k\\plain_text\\1.0.0\\152771d7ae284673c3ad7ffdd9b3afc2741f1d00",
    "output_dir": "C:\\codes\\XAI\\attribution_graphs\\results",
    "clt_dir": "C:\\codes\\XAI\\attribution_graphs\\results\\clt",
    "batch_size": 4,
    "seq_length": 128,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "epochs": 3,
    "warmup_steps": 100,
    "max_grad_norm": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "attribution_samples": 20,
    "edge_threshold": 0.01
}

# CLT配置
CLT_CONFIG = {
    "hidden_size": 768,  # 与原始模型相同的隐藏层大小
    "num_layers": 12,   # 与原始模型相同的层数
    "features_per_layer": 10000,  # 每层特征数量
    "sparsity_lambda": 1e-3,  # 稀疏性惩罚系数
    "sparsity_c": 10.0,  # 稀疏性惩罚参数c
    "learning_rate": 1e-4,  # 学习率
    "weight_decay": 0.01,  # 权重衰减
    "batch_size": 4,  # 批次大小
    "epochs": 5,  # 训练轮数
    "warmup_steps": 100,  # 预热步数
    "max_grad_norm": 1.0,  # 梯度裁剪
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "seed": 42,  # 随机种子
    "save_interval": 1,  # 保存间隔（轮数）
}

def train_clt(config, clt_config):
    """
    训练CLT模型
    
    Args:
        config: 主配置
        clt_config: CLT配置
        
    Returns:
        clt: 训练好的CLT模型
    """
    print("开始训练CLT模型...")
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["clt_dir"], exist_ok=True)
    
    # 设置随机种子
    set_seed(clt_config["seed"])
    
    # 加载数据
    dataloader = get_dataloader(
        config["data_path"],
        config["model_path"],
        clt_config["batch_size"],
        config["seq_length"]
    )
    
    # 加载原始模型
    device = clt_config["device"]
    original_model = load_model(config["model_path"], device)
    
    # 获取模型的隐藏层大小和层数
    if hasattr(original_model.config, "hidden_size"):
        clt_config["hidden_size"] = original_model.config.hidden_size
    if hasattr(original_model.config, "num_hidden_layers"):
        clt_config["num_layers"] = original_model.config.num_hidden_layers
    
    # 创建CLT模型
    clt = CrossLayerTranscoder(clt_config).to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(
        clt.parameters(),
        lr=clt_config["learning_rate"],
        weight_decay=clt_config["weight_decay"]
    )
    
    # 训练循环
    for epoch in range(clt_config["epochs"]):
        print(f"Epoch {epoch+1}/{clt_config['epochs']}")
        clt.train()
        total_loss = 0
        total_mse_loss = 0
        total_sparsity_loss = 0
        num_batches = 0
        
        # 稀疏性惩罚系数固定为1e-3，不再随epoch线性增加
        # clt.sparsity_lambda = 1e-3 (已在初始化时设置)
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            data, _ = batch
            data = data.to(device)
            
            # 收集原始模型的激活值
            residual_stream_activations = []
            mlp_outputs = []
            hooks = []
            
            # 注册钩子以捕获残差流激活值和MLP输出
            def capture_residual_stream(module, input, output, layer_idx):
                residual_stream_activations.append(output.detach())
            
            def capture_mlp_output(module, input, output, layer_idx):
                mlp_outputs.append(output.detach())
            
            # 注册钩子
            for i, layer in enumerate(original_model.layers):
                hooks.append(layer.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                ))
                hooks.append(layer.mlp.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                ))
            
            # 前向传播原始模型
            with torch.no_grad():
                original_model(data)
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 前向传播CLT
            optimizer.zero_grad()
            reconstructed_outputs, feature_activations = clt(residual_stream_activations)
            
            # 计算损失
            loss, mse_loss, sparsity_loss = clt.compute_loss(
                reconstructed_outputs, mlp_outputs, feature_activations
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(clt.parameters(), clt_config["max_grad_norm"])
            
            # 更新参数
            optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            num_batches += 1
            
            # 打印损失
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
            
            # 为了快速测试，只训练少量批次
            if batch_idx >= clt_config.get("max_batches_per_epoch", 50):
                break
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_sparsity_loss = total_sparsity_loss / num_batches
        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg MSE Loss: {avg_mse_loss:.4f}, Avg Sparsity Loss: {avg_sparsity_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % clt_config["save_interval"] == 0:
            save_path = os.path.join(config["clt_dir"], f"clt_epoch_{epoch+1}.pt")
            torch.save({
                "model_state_dict": clt.state_dict(),
                "config": clt_config
            }, save_path)
            print(f"CLT模型已保存到: {save_path}")
    
    # 保存最终模型
    save_path = os.path.join(config["clt_dir"], f"clt_final.pt")
    torch.save({
        "model_state_dict": clt.state_dict(),
        "config": clt_config
    }, save_path)
    print(f"最终CLT模型已保存到: {save_path}")
    
    return clt

# 在load_clt函数中添加model_type参数
def load_clt(clt_path, device="cuda"):
    """加载CLT模型"""
    checkpoint = torch.load(clt_path, map_location=device)
    clt_config = checkpoint["config"]
    model_type = checkpoint.get("model_type", "qwen2")  # 获取保存的模型类型，默认为qwen2
    
    clt = CrossLayerTranscoder(clt_config).to(device)
    clt.load_state_dict(checkpoint["clt_state_dict"])
    clt.eval()
    
    return clt, clt_config, model_type

# 在analyze_clt函数中传递model_type
def analyze_clt(model, clt, dataloader, device, config):
    """分析CLT模型性能"""
    # 获取模型类型
    model_type = CONFIG.get("model_type", "qwen2")
    
    # 创建替换模型
    replacement_model = ReplacementModel(model, clt, model_type=model_type).to(device)
    
    # 其余代码保持不变...

# 在compute_attribution_graphs函数中传递model_type
def compute_attribution_graphs(model, dataloader, device, config, clt=None):
    """计算并保存归因图"""
    # 获取模型类型
    model_type = CONFIG.get("model_type", "qwen2")
    
    if clt is not None:
        # 使用CLT创建替换模型
        replacement_model = ReplacementModel(model, clt, model_type=model_type).to(device)
        # ...
    
    # 其余代码保持不变...

def build_local_replacement_model(original_model, clt, input_ids, attention_mask=None):
    """
    构建局部替换模型
    
    Args:
        original_model: 原始模型
        clt: CLT模型
        input_ids: 输入token IDs
        attention_mask: 注意力掩码
        
    Returns:
        local_replacement_model: 局部替换模型
    """
    device = next(original_model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    local_replacement_model = LocalReplacementModel(
        original_model, clt, input_ids, attention_mask
    ).to(device)
    
    return local_replacement_model

def compute_attribution_graphs(model, dataloader, device, config, clt=None):
    """
    计算并保存归因图
    
    Args:
        model: 要分析的模型
        dataloader: 数据加载器
        device: 设备
        config: 配置
        clt: CLT模型（如果使用CLT）
        
    Returns:
        edge_weights: 归因图的边权重
    """
    model.eval()
    
    attribution_config = {
        "edge_threshold": config["edge_threshold"]
    }
    
    # 根据是否使用CLT选择归因图计算器
    if clt is not None:
        attribution_calculator = AttributionGraphCLT(model, attribution_config)
    else:
        from attribution import AttributionGraph
        attribution_calculator = AttributionGraph(model, attribution_config)
    
    # 收集样本
    inputs = []
    targets = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if i >= config["attribution_samples"]:
                break
                
            inputs.append(data[0].to(device))
            # 随机选择一个目标token
            vocab_size = model.config.vocab_size
            target_idx = np.random.randint(0, vocab_size)
            targets.append(target_idx)
    
    # 构建归因图
    print("计算归因图...")
    edge_weights = attribution_calculator.build_attribution_graph(inputs, targets)
    print(f"归因图计算完成，共有{len(edge_weights)}条边")
    
    # 可视化归因图
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 根据是否使用CLT生成不同的输出文件名
    if clt is not None:
        output_path = os.path.join(config["output_dir"], "clt_attribution_graph.png")
    else:
        output_path = os.path.join(config["output_dir"], "attribution_graph.png")
        
    attribution_calculator.visualize_graph(edge_weights, output_path)
    print(f"归因图已保存到: {output_path}")
    
    return edge_weights

def main():
    parser = argparse.ArgumentParser(description="训练CLT模型并构建归因图")
    parser.add_argument("--mode", type=str, choices=["train", "analyze", "both"], default="both",
                        help="运行模式：train（训练CLT）、analyze（构建归因图）或both（两者都执行）")
    parser.add_argument("--clt_path", type=str, default=None,
                        help="CLT模型路径（用于分析模式）")
    parser.add_argument("--use_original", action="store_true",
                        help="使用原始模型而不是CLT构建归因图")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["clt_dir"], exist_ok=True)
    
    # 设置随机种子
    set_seed(CONFIG["seed"])
    
    # 加载数据
    dataloader = get_dataloader(
        CONFIG["data_path"],
        CONFIG["model_path"],
        CONFIG["batch_size"],
        CONFIG["seq_length"]
    )
    
    # 加载原始模型
    device = CONFIG["device"]
    original_model = load_model(CONFIG["model_path"], device)
    
    clt = None
    
    # 训练CLT模型
    if args.mode in ["train", "both"]:
        clt = train_clt(CONFIG, CLT_CONFIG)
    
    # 构建归因图
    if args.mode in ["analyze", "both"]:
        if args.use_original:
            # 使用原始模型构建归因图
            print("使用原始模型构建归因图...")
            compute_attribution_graphs(original_model, dataloader, device, CONFIG)
        else:
            # 加载CLT模型（如果需要）
            if clt is None:
                if args.clt_path is None:
                    # 使用最新的CLT模型
                    clt_dir = CONFIG["clt_dir"]
                    clt_files = [f for f in os.listdir(clt_dir) if f.endswith(".pt")]
                    if not clt_files:
                        raise ValueError("未找到CLT模型文件")
                    clt_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]) if "_" in x else float('inf'))
                    clt_path = os.path.join(clt_dir, clt_files[-1])
                else:
                    clt_path = args.clt_path
                
                clt, clt_config = load_clt(clt_path, device)
            
            # 获取一个样本用于构建局部替换模型
            for batch in dataloader:
                data, _ = batch
                break
            
            # 构建局部替换模型
            local_replacement_model = build_local_replacement_model(
                original_model, clt, data[0].unsqueeze(0)
            )
            
            # 使用局部替换模型构建归因图
            print("使用CLT局部替换模型构建归因图...")
            compute_attribution_graphs(local_replacement_model, dataloader, device, CONFIG, clt)

if __name__ == "__main__":
    main()