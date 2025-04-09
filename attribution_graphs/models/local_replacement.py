import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class LocalReplacementModel(nn.Module):
    """
    局部替换模型，用于特定提示的归因图分析
    1. 替换MLP层为CLT
    2. 使用原始模型在特定提示上的注意力模式和归一化分母
    3. 添加误差调整项
    """
    def __init__(self, original_model, clt, prompt_input_ids, prompt_attention_mask=None):
        super().__init__()
        self.original_model = original_model
        self.clt = clt
        self.config = original_model.config
        self.device = next(original_model.parameters()).device
        
        # 存储原始模型在特定提示上的激活值和注意力模式
        self.cached_attention_patterns = {}
        self.cached_norm_denominators = {}
        self.cached_mlp_outputs = {}
        self.cached_residual_stream = {}
        self.error_terms = {}
        
        # 在特定提示上运行原始模型并缓存所需信息
        self._cache_original_model_activations(prompt_input_ids, prompt_attention_mask)
        
        # 计算误差调整项
        self._compute_error_terms()
    
    def _cache_original_model_activations(self, input_ids, attention_mask=None):
        """
        在特定提示上运行原始模型并缓存注意力模式、归一化分母和MLP输出
        """
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 存储激活值的钩子
        hooks = []
        
        # 捕获残差流激活值
        def capture_residual_stream(module, input, output, layer_idx):
            self.cached_residual_stream[layer_idx] = output.detach()
        
        # 捕获MLP输出
        def capture_mlp_output(module, input, output, layer_idx):
            self.cached_mlp_outputs[layer_idx] = output.detach()
        
        # 捕获注意力模式
        def capture_attention_pattern(module, input, output, layer_idx):
            # 对于Qwen模型，注意力输出通常包含注意力权重
            if isinstance(output, tuple) and len(output) > 1:
                # 假设第二个元素是注意力权重
                self.cached_attention_patterns[layer_idx] = output[1].detach()
        
        # 捕获归一化分母
        def capture_norm_denominator(module, input, output, layer_idx):
            # 对于LayerNorm，我们需要缓存归一化分母
            # 通常是输入的方差+epsilon
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0]
                # 计算方差
                mean = inp.mean(dim=-1, keepdim=True)
                var = ((inp - mean) ** 2).mean(dim=-1, keepdim=True)
                self.cached_norm_denominators[layer_idx] = (var + module.eps).sqrt().detach()
        
        # 注册钩子
        for i, layer in enumerate(self.original_model.layers):
            # 残差流激活值
            hooks.append(layer.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
            ))
            
            # MLP输出
            hooks.append(layer.mlp.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
            ))
            
            # 注意力模式
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_attention_pattern(module, input, output, layer_idx)
                ))
            elif hasattr(layer, 'attention'):
                hooks.append(layer.attention.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_attention_pattern(module, input, output, layer_idx)
                ))
            
            # 归一化分母
            if hasattr(layer, 'ln_1'):
                hooks.append(layer.ln_1.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_norm_denominator(module, input, output, layer_idx)
                ))
            elif hasattr(layer, 'input_layernorm'):
                hooks.append(layer.input_layernorm.register_forward_hook(
                    lambda module, input, output, layer_idx=i: capture_norm_denominator(module, input, output, layer_idx)
                ))
        
        # 前向传播原始模型
        with torch.no_grad():
            outputs = self.original_model(input_ids, attention_mask=attention_mask)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 确保所有层都被缓存
        num_layers = len(self.original_model.layers)
        for i in range(num_layers):
            if i not in self.cached_residual_stream:
                print(f"警告：第{i}层的残差流激活值未被缓存")
            if i not in self.cached_mlp_outputs:
                print(f"警告：第{i}层的MLP输出未被缓存")
        
        return outputs
    
    def _compute_error_terms(self):
        """
        计算误差调整项：原始MLP输出与CLT输出之间的差异
        """
        # 获取残差流激活值列表
        num_layers = len(self.cached_residual_stream)
        residual_stream_list = [self.cached_residual_stream[i] for i in range(num_layers)]
        
        # 使用CLT计算重建的MLP输出
        with torch.no_grad():
            reconstructed_outputs, _ = self.clt(residual_stream_list)
        
        # 计算误差调整项
        for layer in range(num_layers):
            if layer in self.cached_mlp_outputs and layer < len(reconstructed_outputs):
                self.error_terms[layer] = self.cached_mlp_outputs[layer] - reconstructed_outputs[layer]
            else:
                print(f"警告：无法为第{layer}层计算误差调整项")
    
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        """
        前向传播，使用缓存的注意力模式和添加误差调整项
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            use_cache: 是否使用缓存（用于生成）
            
        Returns:
            outputs: 模型输出
        """
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 存储当前前向传播的激活值
        current_residual_stream = {}
        current_layer_outputs = {}
        
        # 存储激活值的钩子
        hooks = []
        
        # 捕获残差流激活值
        def capture_current_residual(module, input, output, layer_idx):
            current_residual_stream[layer_idx] = output
        
        # 注册钩子
        for i, layer in enumerate(self.original_model.layers):
            hooks.append(layer.register_forward_hook(
                lambda module, input, output, layer_idx=i: capture_current_residual(module, input, output, layer_idx)
            ))
        
        # 获取输入嵌入
        if hasattr(self.original_model, 'wte'):
            # GPT风格模型
            hidden_states = self.original_model.wte(input_ids)
        elif hasattr(self.original_model, 'embed_tokens'):
            # BERT/RoBERTa风格模型
            hidden_states = self.original_model.embed_tokens(input_ids)
        elif hasattr(self.original_model, 'embeddings'):
            # 通用嵌入
            hidden_states = self.original_model.embeddings(input_ids)
        else:
            # 尝试直接运行模型的前向传播获取嵌入
            # 这可能需要根据具体模型架构调整
            raise ValueError("无法确定模型的嵌入层")
        
        # 前向传播，但使用CLT替换MLP输出
        for i, layer in enumerate(self.original_model.layers):
            # 应用注意力层，使用缓存的注意力模式
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn = layer.attention
            else:
                raise ValueError(f"无法找到第{i}层的注意力模块")
            
            # 应用输入归一化
            if hasattr(layer, 'ln_1'):
                norm = layer.ln_1
            elif hasattr(layer, 'input_layernorm'):
                norm = layer.input_layernorm
            else:
                raise ValueError(f"无法找到第{i}层的输入归一化模块")
            
            # 应用归一化
            normalized = norm(hidden_states)
            
            # 应用注意力（使用原始注意力模式）
            # 注意：这里简化了注意力计算，实际实现可能需要根据具体模型架构调整
            attn_output = attn(normalized, attention_mask=attention_mask)[0]
            
            # 残差连接
            hidden_states = hidden_states + attn_output
            
            # 应用MLP层，但使用CLT输出 + 误差调整项
            # 首先获取当前残差流激活值
            current_residual = hidden_states
            
            # 使用CLT计算MLP输出
            with torch.no_grad():
                # 这里简化了实现，实际上需要为每个位置单独计算CLT输出
                # 使用当前残差流激活值计算CLT输出
                clt_input = [current_residual]
                clt_output, _ = self.clt(clt_input)
                
                # 添加误差调整项
                if i in self.error_terms:
                    adjusted_output = clt_output[0] + self.error_terms[i]
                else:
                    adjusted_output = clt_output[0]
            
            # 残差连接
            hidden_states = hidden_states + adjusted_output
            
            # 存储当前层输出
            current_layer_outputs[i] = hidden_states
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 应用最终的归一化
        if hasattr(self.original_model, 'ln_f'):
            hidden_states = self.original_model.ln_f(hidden_states)
        elif hasattr(self.original_model, 'final_layernorm'):
            hidden_states = self.original_model.final_layernorm(hidden_states)
        
        # 应用语言模型头
        if hasattr(self.original_model, 'lm_head'):
            logits = self.original_model.lm_head(hidden_states)
        else:
            # 尝试找到输出层
            raise ValueError("无法确定模型的输出层")
        
        # 返回与原始模型相同格式的输出
        return logits