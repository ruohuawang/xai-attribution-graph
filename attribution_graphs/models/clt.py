import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class JumpReLU(nn.Module):
    """
    JumpReLU激活函数，用于CLT中的特征激活
    论文中使用JumpReLU + 直通梯度估计器来实现稀疏激活
    论文中设置阈值为0.01
    """
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))
    
    def forward(self, x):
        # 前向传播时使用阈值
        mask = (x > self.threshold).float()
        # 使用直通梯度估计器
        if self.training:
            # 训练时使用STE (Straight-Through Estimator)
            return x * mask + x.detach() * (1 - mask)
        else:
            # 推理时直接应用阈值
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
        self.features_per_layer = config.get("features_per_layer", 10000)  # 论文中设置为10000
        self.sparsity_lambda = 1e-3  # 论文中固定为1e-3
        self.sparsity_c = 10.0  # 论文中固定为10.0
        
        # 检查是否需要输入适配器
        self.input_dim = config.get("actual_hidden_size", self.hidden_size)
        self.use_input_adapter = (self.input_dim != self.hidden_size)
        
        # 为每一层创建编码器和解码器
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleDict()
        self.activations = nn.ModuleList()
        
        # 如果需要，创建输入适配器 - 修复维度顺序
        if self.use_input_adapter:
            self.input_adapters = nn.ModuleList()
            for _ in range(self.num_layers):
                # 从input_dim到hidden_size的转换
                self.input_adapters.append(nn.Linear(self.input_dim, self.hidden_size))
        
        # 初始化每层的编码器
        for layer in range(self.num_layers):
            # 编码器：从残差流到特征
            self.encoders.append(nn.Linear(self.hidden_size, self.features_per_layer))
            # 激活函数
            self.activations.append(JumpReLU())
            
            # 修改：为每一层创建从1到ℓ层的解码器
            # 即第ℓ层接收来自第1到第ℓ层的特征输入
            for source_layer in range(layer + 1):  # 包括当前层自身
                decoder_name = f"{source_layer}_to_{layer}"
                self.decoders[decoder_name] = nn.Linear(self.features_per_layer, self.hidden_size)
        
        # 如果需要，创建输出适配器
        if self.use_input_adapter:
            self.output_adapters = nn.ModuleList()
            for _ in range(self.num_layers):
                self.output_adapters.append(nn.Linear(self.hidden_size, self.input_dim))
    
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
        
        # 打印输入维度以进行调试
        #print(f"输入维度: {residual_stream_activations[0].shape}")
        
        # 存储每层特征的激活值
        feature_activations = []
        # 存储重建的MLP输出
        reconstructed_mlp_outputs = []
        
        # 首先计算所有层的特征激活值
        for layer in range(min(len(residual_stream_activations), self.num_layers)):
            # 编码：从残差流到特征
            layer_input = residual_stream_activations[layer]
            
            # 如果需要，应用输入适配器
            if self.use_input_adapter:
                # 打印适配前维度
                print(f"第{layer}层适配前维度: {layer_input.shape}")
                layer_input = self.input_adapters[layer](layer_input)
                # 打印适配后维度
                print(f"第{layer}层适配后维度: {layer_input.shape}")
                
            feature_preact = self.encoders[layer](layer_input)
            feature_act = self.activations[layer](feature_preact)
            feature_activations.append(feature_act)
        
        # 然后使用特征激活值重建MLP输出
        for layer in range(min(len(residual_stream_activations), self.num_layers)):
            # 初始化该层的重建输出
            reconstructed_output = torch.zeros(
                batch_size, seq_len, self.hidden_size, 
                device=residual_stream_activations[0].device
            )
            
            # 解码：使用从第1层到第ℓ层的特征重建第ℓ层的MLP输出
            for source_layer in range(layer + 1):  # 包括当前层自身
                decoder_name = f"{source_layer}_to_{layer}"
                decoder_output = self.decoders[decoder_name](feature_activations[source_layer])
                reconstructed_output += decoder_output
            
            # 如果需要，应用输出适配器
            if self.use_input_adapter:
                reconstructed_output = self.output_adapters[layer](reconstructed_output)
                
            reconstructed_mlp_outputs.append(reconstructed_output)
        
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
        for i, (recon, target) in enumerate(zip(reconstructed_outputs, target_outputs)):
            # 确保维度匹配
            if recon.shape[-1] != target.shape[-1]:
                print(f"警告：第{i}层的输出维度不匹配 - recon: {recon.shape}, target: {target.shape}")
                continue
                
            mse_loss += F.mse_loss(recon, target)
        
        # 计算稀疏性惩罚
        sparsity_loss = 0
        
        # 对每层的特征计算稀疏性惩罚
        for layer in range(self.num_layers):
            feature_act = feature_activations[layer]
            batch_size, seq_len, num_features = feature_act.shape
            
            # 对每个特征计算稀疏性惩罚
            for feature_idx in range(num_features):
                # 收集该特征所有解码器的权重
                all_decoder_weights = []
                
                # 查找所有使用该层特征的解码器
                for target_layer in range(layer, self.num_layers):
                    decoder_name = f"{layer}_to_{target_layer}"
                    if decoder_name in self.decoders:
                        # 获取解码器权重的特定列（对应特定特征）
                        decoder_weight_col = self.decoders[decoder_name].weight[:, feature_idx]
                        all_decoder_weights.append(decoder_weight_col)
                
                # 连接所有解码器权重
                if all_decoder_weights:
                    concatenated_weights = torch.cat(all_decoder_weights)
                    # 计算连接权重的范数
                    weight_norm = torch.norm(concatenated_weights)
                    
                    # 计算该特征的激活值
                    feature_activation = feature_act[:, :, feature_idx]
                    
                    # 计算tanh(c * ||W_dec,i^l|| * a_i^l)的和
                    sparsity_term = torch.tanh(self.sparsity_c * weight_norm * feature_activation)
                    sparsity_loss += sparsity_term.sum()
        
        # 总损失 - 使用固定的稀疏性惩罚系数
        total_loss = mse_loss + self.sparsity_lambda * sparsity_loss
        
        return total_loss, mse_loss, sparsity_loss

class ReplacementModel(nn.Module):
    """
    使用训练好的CLT替换原始模型的MLP层
    论文中描述的替换模型，用于在推理时替换原始模型的MLP层
    """
    def __init__(self, original_model, clt, model_type=None):
        super().__init__()
        self.original_model = original_model
        self.clt = clt
        self.config = clt.config
        self.model_type = model_type if model_type else "qwen2"  # 默认为qwen2
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播，使用CLT替换原始模型的MLP层
        论文中描述的替换逻辑：在前向传播过程中捕获残差流激活值，
        然后使用CLT计算MLP输出并替换原始MLP输出
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            outputs: 模型输出
        """
        # 存储当前前向传播的残差流激活值
        residual_stream_activations = {}
        
        # 创建一个字典来存储CLT重建的MLP输出
        reconstructed_mlp_outputs = {}
        
        # 注册钩子以捕获残差流激活值
        def capture_residual_stream(module, input, output, layer_idx):
            # 处理输出可能是元组的情况
            if isinstance(output, tuple):
                residual_stream_activations[layer_idx] = output[0]
            else:
                residual_stream_activations[layer_idx] = output
            return output
        
        # 替换MLP输出的钩子函数
        def replace_mlp_output(module, input, output, layer_idx):
            # 如果当前层的残差流激活值已被捕获
            if layer_idx in residual_stream_activations:
                # 如果还没有计算过CLT输出
                if not reconstructed_mlp_outputs:
                    # 将残差流激活值转换为列表格式，供CLT使用
                    max_layer = max(residual_stream_activations.keys())
                    residual_list = [residual_stream_activations.get(i, None) for i in range(max_layer + 1)]
                    # 过滤掉None值
                    residual_list = [r for r in residual_list if r is not None]
                    # 使用CLT计算重建的MLP输出
                    recon_outputs, _ = self.clt(residual_list)
                    # 将重建输出存储到字典中
                    for i, recon_out in enumerate(recon_outputs):
                        reconstructed_mlp_outputs[i] = recon_out
                
                # 返回CLT重建的输出替代原始MLP输出
                if layer_idx in reconstructed_mlp_outputs:
                    # 处理输出可能是元组的情况
                    if isinstance(output, tuple):
                        return (reconstructed_mlp_outputs[layer_idx],) + output[1:]
                    else:
                        return reconstructed_mlp_outputs[layer_idx]
            
            # 如果无法替换，则返回原始输出
            return output
        
        # 注册钩子
        hooks = []
        
        if self.model_type == "qwen2":
            # Qwen模型结构
            for i, layer in enumerate(self.original_model.transformer.h):
                # 捕获残差流激活值
                hooks.append(layer.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                ))
                
                # 替换MLP输出
                hooks.append(layer.mlp.register_forward_hook(
                    lambda module, input, output, layer_idx=i: replace_mlp_output(module, input, output, layer_idx)
                ))
        
        elif self.model_type == "gpt2":
            # GPT-2模型结构
            for i, layer in enumerate(self.original_model.h):
                # 捕获残差流激活值
                hooks.append(layer.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                ))
                
                # 替换MLP输出
                hooks.append(layer.mlp.register_forward_hook(
                    lambda module, input, output, layer_idx=i: replace_mlp_output(module, input, output, layer_idx)
                ))
        
        elif self.model_type == "bert":
            # BERT模型结构
            for i, layer in enumerate(self.original_model.encoder.layer):
                # 捕获残差流激活值
                hooks.append(layer.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                ))
                
                # 替换MLP输出 (BERT中称为intermediate)
                hooks.append(layer.intermediate.register_forward_hook(
                    lambda module, input, output, layer_idx=i: replace_mlp_output(module, input, output, layer_idx)
                ))
        
        else:
            # 通用方法，尝试自动检测模型结构
            if hasattr(self.original_model, 'transformer') and hasattr(self.original_model.transformer, 'h'):
                layers = self.original_model.transformer.h
                mlp_name = 'mlp'
            elif hasattr(self.original_model, 'h'):
                layers = self.original_model.h
                mlp_name = 'mlp'
            elif hasattr(self.original_model, 'encoder') and hasattr(self.original_model.encoder, 'layer'):
                layers = self.original_model.encoder.layer
                mlp_name = 'intermediate'
            else:
                raise ValueError(f"不支持的模型结构: {type(self.original_model).__name__}")
            
            for i, layer in enumerate(layers):
                # 捕获残差流激活值
                hooks.append(layer.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                ))
                
                # 替换MLP输出
                mlp = getattr(layer, mlp_name, None)
                if mlp is not None:
                    hooks.append(mlp.register_forward_hook(
                        lambda module, input, output, layer_idx=i: replace_mlp_output(module, input, output, layer_idx)
                    ))
        
        # 前向传播，钩子函数会自动替换MLP输出
        if self.model_type == "bert" and attention_mask is None and input_ids is not None:
            # BERT需要attention_mask
            attention_mask = torch.ones_like(input_ids)
        
        outputs = self.original_model(input_ids, attention_mask=attention_mask)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return outputs

# 导入LocalReplacementModel
from models.local_replacement import LocalReplacementModel