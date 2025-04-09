import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class JumpReLU(nn.Module):
    """
    JumpReLU激活函数，用于CLT中的特征激活
    论文中使用JumpReLU + 直通梯度估计器来实现稀疏激活
    """
    def __init__(self, threshold=0.03):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
        
    def forward(self, x):
        # 前向传播时使用阈值
        mask = (x > self.threshold).float()
        # 使用直通梯度估计器，带宽参数为1.0
        if self.training:
            return x * mask + x.detach() * (1 - mask)
        else:
            return x * mask

class CrossLayerTranscoder(nn.Module):
    """
    Cross-Layer Transcoder (CLT) 实现
    CLT由分布在L层的特征组成，每层特征从该层的残差流中读取输入，
    并可以向所有后续层提供输出
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.features_per_layer = config["features_per_layer"]
        self.sparsity_lambda = config.get("sparsity_lambda", 1e-3)
        self.sparsity_c = config.get("sparsity_c", 10.0)
        
        # 为每一层创建编码器和解码器
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleDict()
        self.activations = nn.ModuleList()
        
        # 初始化每层的编码器
        for layer in range(self.num_layers):
            # 编码器：从残差流到特征
            self.encoders.append(nn.Linear(self.hidden_size, self.features_per_layer))
            # 激活函数
            self.activations.append(JumpReLU())
            
            # 为当前层创建到所有后续层的解码器
            for target_layer in range(layer, self.num_layers):
                decoder_name = f"{layer}_to_{target_layer}"
                self.decoders[decoder_name] = nn.Linear(self.features_per_layer, self.hidden_size)
    
    def forward(self, residual_stream_activations):
        """
        前向传播，计算CLT的输出
        
        Args:
            residual_stream_activations: 原始模型各层的残差流激活值列表 [batch_size, seq_len, hidden_size]
            
        Returns:
            reconstructed_mlp_outputs: 重建的MLP输出列表
            feature_activations: 各层特征的激活值
        """
        batch_size, seq_len = residual_stream_activations[0].shape[:2]
        
        # 存储每层特征的激活值
        feature_activations = []
        # 存储重建的MLP输出
        reconstructed_mlp_outputs = [torch.zeros_like(act) for act in residual_stream_activations]
        
        # 对每一层进行处理
        for layer in range(self.num_layers):
            # 编码：从残差流到特征
            layer_input = residual_stream_activations[layer]
            feature_preact = self.encoders[layer](layer_input)
            feature_act = self.activations[layer](feature_preact)
            feature_activations.append(feature_act)
            
            # 解码：从特征到MLP输出（对当前层及所有后续层）
            for target_layer in range(layer, self.num_layers):
                decoder_name = f"{layer}_to_{target_layer}"
                decoder_output = self.decoders[decoder_name](feature_act)
                reconstructed_mlp_outputs[target_layer] += decoder_output
        
        return reconstructed_mlp_outputs, feature_activations
    
    def compute_loss(self, reconstructed_outputs, target_outputs, feature_activations):
        """
        计算CLT的损失函数：重建误差 + 稀疏性惩罚
        
        Args:
            reconstructed_outputs: CLT重建的MLP输出
            target_outputs: 原始模型的MLP输出
            feature_activations: CLT各层特征的激活值
            
        Returns:
            total_loss: 总损失
            mse_loss: 重建误差
            sparsity_loss: 稀疏性惩罚
        """
        # 计算重建误差（MSE损失）
        mse_loss = 0
        for recon, target in zip(reconstructed_outputs, target_outputs):
            mse_loss += F.mse_loss(recon, target)
        
        # 计算稀疏性惩罚
        sparsity_loss = 0
        for layer in range(self.num_layers):
            feature_act = feature_activations[layer]
            
            # 计算每个特征的解码器权重范数
            decoder_norms = []
            for target_layer in range(layer, self.num_layers):
                decoder_name = f"{layer}_to_{target_layer}"
                decoder_weights = self.decoders[decoder_name].weight
                decoder_norms.append(torch.norm(decoder_weights, dim=0))
            
            # 连接所有解码器权重范数
            if decoder_norms:
                all_norms = torch.stack(decoder_norms, dim=0).sum(dim=0)
                # 计算tanh(c * ||W_dec,i^l|| * a_i^l)的和
                sparsity_term = torch.tanh(self.sparsity_c * all_norms.unsqueeze(0).unsqueeze(0) * feature_act)
                sparsity_loss += sparsity_term.sum()
        
        # 总损失
        total_loss = mse_loss + self.sparsity_lambda * sparsity_loss
        
        return total_loss, mse_loss, sparsity_loss

class ReplacementModel(nn.Module):
    """
    使用训练好的CLT替换原始模型的MLP层
    """
    def __init__(self, original_model, clt):
        super().__init__()
        self.original_model = original_model
        self.clt = clt
        self.config = clt.config
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播，使用CLT替换原始模型的MLP层
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            outputs: 模型输出
        """
        # 获取原始模型的钩子
        hooks = []
        residual_stream_activations = []
        mlp_outputs = []
        
        # 注册钩子以捕获残差流激活值和MLP输出
        def capture_residual_stream(module, input, output, layer_idx):
            residual_stream_activations.append(output.detach())
        
        def capture_mlp_output(module, input, output, layer_idx):
            mlp_outputs.append(output.detach())
        
        # 注册钩子
        for i, layer in enumerate(self.original_model.layers):
            hooks.append(layer.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
            ))
            hooks.append(layer.mlp.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
            ))
        
        # 前向传播，但在MLP输出处使用CLT的输出
        outputs = self.original_model(input_ids, attention_mask=attention_mask)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 使用CLT计算重建的MLP输出
        reconstructed_outputs, _ = self.clt(residual_stream_activations)
        
        # TODO: 实现完整的替换逻辑，这需要修改原始模型的前向传播
        # 当前实现仅用于演示，实际使用时需要在前向传播过程中替换MLP输出
        
        return outputs

# 导入LocalReplacementModel
from models.local_replacement import LocalReplacementModel