import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Dataset

from data_loader import get_dataloader
from model_loader import load_model
from models.clt import CrossLayerTranscoder, ReplacementModel, LocalReplacementModel
from utils.attribution import AttributionGraph

# 设置随机种子以确保可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# CLT训练配置
CLT_CONFIG = {
    "hidden_size": 512,  # 与原始模型相同的隐藏层大小
    "num_layers": 12,   # 与原始模型相同的层数
    "features_per_layer": 5000,  # 每层特征数量
    "sparsity_lambda": 1e-3,  # 稀疏性惩罚系数
    "sparsity_c": 10.0,  # 稀疏性惩罚参数c
    "learning_rate": 1e-4,  # 学习率
    "weight_decay": 0.01,  # 权重衰减
    "batch_size": 2,  # 批次大小
    "epochs": 10,  # 训练轮数
    "warmup_steps": 100,  # 预热步数
    "max_grad_norm": 1.0,  # 梯度裁剪
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 设备
    "seed": 42,  # 随机种子
    "save_interval": 1,  # 保存间隔（轮数）
}

# 主配置
CONFIG = {
    "model_path": "C:\\codes\\llms\\qwen05b",
    "data_path": "C:\\codes\\XAI\\attribution_graphs\\data\\stas___openwebtext-10k\\plain_text\\1.0.0\\152771d7ae284673c3ad7ffdd9b3afc2741f1d00",
    "output_dir": "C:\\codes\\XAI\\attribution_graphs\\results",
    "clt_dir": "C:\\codes\\XAI\\attribution_graphs\\results\\clt",
    "seq_length": 128,
    "attribution_samples": 20,
    "edge_threshold": 0.01
}

class CLTTrainer:
    """
    CLT训练器，用于训练Cross-Layer Transcoder
    """
    def __init__(self, config, clt_config):
        self.config = config
        self.clt_config = clt_config
        self.device = clt_config["device"]
        
        # 创建输出目录
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["clt_dir"], exist_ok=True)
        
        # 设置随机种子
        set_seed(clt_config["seed"])
        
        # 加载数据
        self.dataloader = get_dataloader(
            config["data_path"],
            config["model_path"],
            clt_config["batch_size"],
            config["seq_length"]
        )
        
        # 加载原始模型
        self.original_model = load_model(config["model_path"], self.device)
        
        # 获取模型的隐藏层大小和层数
        if hasattr(self.original_model.config, "hidden_size"):
            self.clt_config["hidden_size"] = self.original_model.config.hidden_size
        if hasattr(self.original_model.config, "num_hidden_layers"):
            self.clt_config["num_layers"] = self.original_model.config.num_hidden_layers
        
        # 创建CLT模型
        self.clt = CrossLayerTranscoder(self.clt_config).to(self.device)
        
        # 定义优化器
        self.optimizer = optim.AdamW(
            self.clt.parameters(),
            lr=clt_config["learning_rate"],
            weight_decay=clt_config["weight_decay"]
        )
    
    def collect_activations(self, batch):
        """
        收集原始模型的残差流激活值和MLP输出
        
        Args:
            batch: 数据批次
            
        Returns:
            residual_stream_activations: 残差流激活值列表
            mlp_outputs: MLP输出列表
        """
        data, _ = batch
        data = data.to(self.device)
        
        # 存储激活值
        residual_stream_activations = []
        mlp_outputs = []
        hooks = []
        
        # 注册钩子以捕获残差流激活值和MLP输出
        def capture_residual_stream(module, input, output, layer_idx):
            # 处理tuple类型的输出
            if isinstance(output, tuple):
                # 通常第一个元素是隐藏状态
                residual_stream_activations.append(output[0].detach())
            else:
                residual_stream_activations.append(output.detach())
        
        def capture_mlp_output(module, input, output, layer_idx):
            # 处理tuple类型的输出
            if isinstance(output, tuple):
                # 通常第一个元素是MLP输出
                mlp_outputs.append(output[0].detach())
            else:
                mlp_outputs.append(output.detach())
        
        # 注册钩子
        for i, layer in enumerate(self.original_model.model.layers):
            hooks.append(layer.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
            ))
            hooks.append(layer.mlp.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
            ))
        
        # 前向传播
        with torch.no_grad():
            self.original_model(data)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return residual_stream_activations, mlp_outputs
    
    def train_epoch(self, epoch):
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch
            
        Returns:
            avg_loss: 平均损失
            avg_mse_loss: 平均MSE损失
            avg_sparsity_loss: 平均稀疏性损失
        """
        self.clt.train()
        total_loss = 0
        total_mse_loss = 0
        total_sparsity_loss = 0
        num_batches = 0
        
        # 计算当前epoch的稀疏性惩罚系数（线性增加）
        current_sparsity_lambda = self.clt_config["sparsity_lambda"] * (epoch / self.clt_config["epochs"])
        self.clt.sparsity_lambda = current_sparsity_lambda
        
        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.clt_config['epochs']}")):
            # 收集原始模型的激活值
            residual_stream_activations, mlp_outputs = self.collect_activations(batch)
            
            # 前向传播CLT
            self.optimizer.zero_grad()
            reconstructed_outputs, feature_activations = self.clt(residual_stream_activations)
            
            # 计算损失
            loss, mse_loss, sparsity_loss = self.clt.compute_loss(
                reconstructed_outputs, mlp_outputs, feature_activations
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.clt.parameters(), self.clt_config["max_grad_norm"])
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            num_batches += 1
            
            # 打印损失
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, MSE Loss: {mse_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
            
            # 为了快速测试，只训练少量批次
            if batch_idx >= 50:
                break
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        avg_sparsity_loss = total_sparsity_loss / num_batches
        
        return avg_loss, avg_mse_loss, avg_sparsity_loss
    
    def evaluate(self):
        """
        评估CLT的性能
        
        Returns:
            avg_mse_loss: 平均MSE损失
            avg_sparsity: 平均稀疏性（激活的特征比例）
        """
        self.clt.eval()
        total_mse_loss = 0
        total_sparsity = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                # 收集原始模型的激活值
                residual_stream_activations, mlp_outputs = self.collect_activations(batch)
                
                # 前向传播CLT
                reconstructed_outputs, feature_activations = self.clt(residual_stream_activations)
                
                # 计算MSE损失
                mse_loss = 0
                for recon, target in zip(reconstructed_outputs, mlp_outputs):
                    mse_loss += nn.functional.mse_loss(recon, target).item()
                
                # 计算稀疏性（激活的特征比例）
                sparsity = 0
                for feature_act in feature_activations:
                    active_features = (feature_act > 0).float().mean().item()
                    sparsity += active_features
                sparsity /= len(feature_activations)
                
                # 累计
                total_mse_loss += mse_loss
                total_sparsity += sparsity
                num_batches += 1
                
                # 为了快速测试，只评估少量批次
                if batch_idx >= 10:
                    break
        
        # 计算平均值
        avg_mse_loss = total_mse_loss / num_batches
        avg_sparsity = total_sparsity / num_batches
        
        return avg_mse_loss, avg_sparsity
    
    def save_model(self, epoch):
        """
        保存CLT模型
        
        Args:
            epoch: 当前epoch
        """
        save_path = os.path.join(self.config["clt_dir"], f"clt_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": self.clt.state_dict(),
            "config": self.clt_config
        }, save_path)
        print(f"CLT模型已保存到: {save_path}")
    
    def train(self):
        """
        训练CLT模型
        """
        print("开始训练CLT模型...")
        print(f"CLT配置: {self.clt_config}")
        
        for epoch in range(self.clt_config["epochs"]):
            # 训练一个epoch
            avg_loss, avg_mse_loss, avg_sparsity_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Avg MSE Loss: {avg_mse_loss:.4f}, Avg Sparsity Loss: {avg_sparsity_loss:.4f}")
            
            # 评估
            eval_mse_loss, eval_sparsity = self.evaluate()
            print(f"Evaluation - MSE Loss: {eval_mse_loss:.4f}, Sparsity: {eval_sparsity:.4f}")
            
            # 保存模型
            if (epoch + 1) % self.clt_config["save_interval"] == 0:
                self.save_model(epoch)
        
        # 保存最终模型
        self.save_model(self.clt_config["epochs"] - 1)
        print("CLT模型训练完成！")

def build_replacement_model(original_model, clt, device):
    """
    构建替换模型
    
    Args:
        original_model: 原始模型
        clt: 训练好的CLT
        device: 设备
        
    Returns:
        replacement_model: 替换模型
    """
    replacement_model = ReplacementModel(original_model, clt).to(device)
    return replacement_model

def build_local_replacement_model(original_model, clt, input_ids, attention_mask, device):
    """
    构建局部替换模型
    
    Args:
        original_model: 原始模型
        clt: 训练好的CLT
        input_ids: 输入token IDs
        attention_mask: 注意力掩码
        device: 设备
        
    Returns:
        local_replacement_model: 局部替换模型
    """
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    local_replacement_model = LocalReplacementModel(
        original_model, clt, input_ids, attention_mask
    ).to(device)
    
    return local_replacement_model

def compute_attribution_graph(local_replacement_model, input_ids, target_idx, config):
    """
    计算归因图
    
    Args:
        local_replacement_model: 局部替换模型
        input_ids: 输入token IDs
        target_idx: 目标token索引
        config: 配置
        
    Returns:
        edge_weights: 归因图的边权重
    """
    attribution_config = {
        "edge_threshold": config["edge_threshold"]
    }
    
    attribution_calculator = AttributionGraph(local_replacement_model, attribution_config)
    edge_weights = attribution_calculator.build_attribution_graph([input_ids], [target_idx])
    
    return edge_weights

def main():
    parser = argparse.ArgumentParser(description="训练CLT模型并构建归因图")
    parser.add_argument("--train", action="store_true", help="训练CLT模型")
    parser.add_argument("--analyze", action="store_true", help="使用训练好的CLT构建归因图")
    parser.add_argument("--clt_path", type=str, default=None, help="CLT模型路径（用于分析）")
    args = parser.parse_args()
    
    if args.train:
        # 训练CLT模型
        trainer = CLTTrainer(CONFIG, CLT_CONFIG)
        trainer.train()
    
    if args.analyze:
        # 加载原始模型
        device = CLT_CONFIG["device"]
        original_model = load_model(CONFIG["model_path"], device)
        
        # 加载CLT模型
        if args.clt_path is None:
            # 使用最新的CLT模型
            clt_dir = CONFIG["clt_dir"]
            clt_files = [f for f in os.listdir(clt_dir) if f.endswith(".pt")]
            if not clt_files:
                raise ValueError("未找到CLT模型文件")
            clt_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            clt_path = os.path.join(clt_dir, clt_files[-1])
        else:
            clt_path = args.clt_path
        
        print(f"加载CLT模型: {clt_path}")
        checkpoint = torch.load(clt_path)
        clt_config = checkpoint["config"]
        
        # 创建CLT模型
        clt = CrossLayerTranscoder(clt_config).to(device)
        clt.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载数据
        dataloader = get_dataloader(
            CONFIG["data_path"],
            CONFIG["model_path"],
            1,  # 批次大小为1
            CONFIG["seq_length"]
        )
        
        # 获取一个样本
        for batch in dataloader:
            data, _ = batch
            break
        
        # 构建局部替换模型
        local_replacement_model = build_local_replacement_model(
            original_model, clt, data, None, device
        )
        
        # 计算归因图
        print("计算归因图...")
        # 随机选择一个目标token
        vocab_size = original_model.config.vocab_size
        target_idx = np.random.randint(0, vocab_size)
        
        edge_weights = compute_attribution_graph(
            local_replacement_model, data[0], target_idx, CONFIG
        )
        
        # 可视化归因图
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
        output_path = os.path.join(CONFIG["output_dir"], "clt_attribution_graph.png")
        
        attribution_calculator = AttributionGraph(local_replacement_model, {"edge_threshold": CONFIG["edge_threshold"]})
        attribution_calculator.visualize_graph(edge_weights, output_path)
        
        print(f"归因图已保存到: {output_path}")

if __name__ == "__main__":
    main()