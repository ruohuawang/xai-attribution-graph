import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx
import os

class AttributionGraphCLT:
    """
    基于CLT的归因图计算
    实现论文中的归因图计算方法，使用局部替换模型
    """
    def __init__(self, model, config):
        """
        初始化归因图计算器
        
        Args:
            model: 要分析的模型（局部替换模型）
            config: 配置参数
        """
        self.model = model
        self.config = config
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        self.feature_activations = {}
        self.feature_gradients = {}
        
    def _register_hooks(self):
        """注册前向和后向钩子以捕获激活和梯度"""
        def _save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
            
        def _save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        def _save_feature_activation(layer_idx):
            def hook(module, input, output):
                self.feature_activations[f"feature_layer_{layer_idx}"] = output.detach()
            return hook
        
        def _save_feature_gradient(layer_idx):
            def hook(module, grad_input, grad_output):
                self.feature_gradients[f"feature_layer_{layer_idx}"] = grad_output[0].detach()
            return hook
        
        # 为模型中的每个层注册钩子
        for name, module in self.model.named_modules():
            # 针对Qwen模型的特定层
            if isinstance(module, nn.Linear) or "attention" in name.lower():
                self.hooks.append(module.register_forward_hook(_save_activation(name)))
                self.hooks.append(module.register_full_backward_hook(_save_gradient(name)))
        
        # 为CLT的特征激活注册钩子
        if hasattr(self.model, 'clt'):
            for i, activation in enumerate(self.model.clt.activations):
                self.hooks.append(activation.register_forward_hook(_save_feature_activation(i)))
                self.hooks.append(activation.register_full_backward_hook(_save_feature_gradient(i)))
    
    def _remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_attribution(self, inputs, target_idx):
        """
        计算输入到目标输出的归因
        
        Args:
            inputs: 模型输入
            target_idx: 目标输出索引
            
        Returns:
            包含每层归因的字典
        """
        self._register_hooks()
        self.activations = {}
        self.gradients = {}
        self.feature_activations = {}
        self.feature_gradients = {}
        
        # 前向传播
        self.model.zero_grad()
        outputs = self.model(inputs)
        
        # 获取logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        # 创建目标的one-hot向量
        target = torch.zeros_like(logits)
        batch_size, seq_len, vocab_size = logits.shape
        
        # 确保target_idx在有效范围内
        valid_target_idx = target_idx % vocab_size
        
        # 对于每个样本的最后一个token位置设置目标
        for i in range(batch_size):
            target[i, -1, valid_target_idx] = 1.0
        
        # 反向传播
        logits.backward(gradient=target)
        
        # 计算归因
        attributions = {}
        
        # 处理普通层的归因
        for name in self.activations:
            if name in self.gradients:
                # 计算激活与梯度的乘积
                act = self.activations[name]
                grad = self.gradients[name]
                
                # 确保形状匹配
                if act.shape == grad.shape:
                    attribution = act * grad
                    attributions[name] = attribution.sum(dim=0)
        
        # 处理CLT特征的归因
        for name in self.feature_activations:
            if name in self.feature_gradients:
                act = self.feature_activations[name]
                grad = self.feature_gradients[name]
                
                if act.shape == grad.shape:
                    attribution = act * grad
                    attributions[name] = attribution.sum(dim=0)
        
        self._remove_hooks()
        return attributions
    
    def build_attribution_graph(self, inputs, target_indices):
        """
        构建多个样本的归因图
        
        Args:
            inputs: 输入样本列表
            target_indices: 每个样本的目标索引
            
        Returns:
            表示归因图的边权重字典
        """
        all_attributions = []
        
        for input_tensor, target_idx in zip(inputs, target_indices):
            attribution = self.compute_attribution(input_tensor.unsqueeze(0), target_idx)
            all_attributions.append(attribution)
        
        # 聚合所有样本的归因
        edge_weights = {}
        for attribution in all_attributions:
            for src_name, src_attr in attribution.items():
                for dst_name, dst_attr in attribution.items():
                    if src_name != dst_name:
                        # 计算两层之间的归因关系
                        edge_key = (src_name, dst_name)
                        
                        # 确保形状匹配，如果不匹配则跳过
                        if src_attr.shape != dst_attr.shape:
                            continue
                            
                        correlation = torch.sum(src_attr * dst_attr).item()
                        
                        if edge_key in edge_weights:
                            edge_weights[edge_key] += correlation
                        else:
                            edge_weights[edge_key] = correlation
        
        # 归一化边权重
        total_weight = sum(abs(w) for w in edge_weights.values())
        if total_weight > 0:
            for edge in edge_weights:
                edge_weights[edge] /= total_weight
        
        # 过滤小于阈值的边
        threshold = self.config["edge_threshold"]
        edge_weights = {k: v for k, v in edge_weights.items() if abs(v) > threshold}
        
        return edge_weights
    
    def visualize_graph(self, edge_weights, output_path):
        """
        可视化归因图
        
        Args:
            edge_weights: 边权重字典
            output_path: 输出图像路径
        """
        G = nx.DiGraph()
        
        # 添加节点和边
        for (src, dst), weight in edge_weights.items():
            # 简化节点名称以便于可视化
            src_simple = src.split('.')[-2] if '.' in src else src
            dst_simple = dst.split('.')[-2] if '.' in dst else dst
            
            G.add_edge(src_simple, dst_simple, weight=weight)
        
        # 设置节点位置
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制图形
        plt.figure(figsize=(15, 12))
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.8)
        
        # 绘制边，根据权重设置宽度和颜色
        edges = nx.draw_networkx_edges(
            G, pos, 
            width=[abs(G[u][v]['weight']) * 10 for u, v in G.edges()],
            edge_color=[G[u][v]['weight'] for u, v in G.edges()],
            edge_cmap=plt.cm.RdBu_r,
            alpha=0.7
        )
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=10)
        
        plt.title("Attribution Graph (CLT)")
        plt.axis('off')
        
        # 修复颜色条问题 - 正确设置颜色条
        if edges is not None and len(G.edges()) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r)
            sm.set_array([G[u][v]['weight'] for u, v in G.edges()])
            plt.colorbar(sm)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"归因图已保存到: {output_path}")