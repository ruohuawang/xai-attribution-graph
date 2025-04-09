import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class AttributionGraph:
    """
    实现论文中的归因图计算
    """
    def __init__(self, model: nn.Module, config: Dict):
        """
        初始化归因图计算器
        
        Args:
            model: 要分析的模型
            config: 配置参数
        """
        self.model = model
        self.config = config
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        
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
        
        # 为模型中的每个层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                self.hooks.append(module.register_forward_hook(_save_activation(name)))
                self.hooks.append(module.register_backward_hook(_save_gradient(name)))
    
    def _remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_attribution(self, inputs: torch.Tensor, target_idx: int) -> Dict[str, torch.Tensor]:
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
        
        # 前向传播
        self.model.zero_grad()
        outputs = self.model(inputs)
        
        # 创建目标的one-hot向量
        target = torch.zeros_like(outputs)
        target[:, target_idx] = 1.0
        
        # 反向传播
        outputs.backward(gradient=target)
        
        # 计算归因
        attributions = {}
        for name in self.activations:
            if name in self.gradients:
                # 计算激活与梯度的乘积
                attribution = self.activations[name] * self.gradients[name]
                attributions[name] = attribution.sum(dim=0)
        
        self._remove_hooks()
        return attributions
    
    def build_attribution_graph(self, 
                               inputs: List[torch.Tensor], 
                               target_indices: List[int]) -> Dict[Tuple[str, str], float]:
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
    
    def visualize_graph(self, edge_weights: Dict[Tuple[str, str], float], output_path: str):
        """
        可视化归因图
        
        Args:
            edge_weights: 边权重字典
            output_path: 输出图像路径
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.DiGraph()
            
            # 添加节点和边
            for (src, dst), weight in edge_weights.items():
                G.add_edge(src, dst, weight=weight)
            
            # 设置节点位置
            pos = nx.spring_layout(G)
            
            # 绘制图形
            plt.figure(figsize=(12, 10))
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.8)
            
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
            
            plt.title("Attribution Graph")
            plt.axis('off')
            plt.colorbar(edges)
            plt.savefig(output_path)
            plt.close()
            
        except ImportError:
            print("需要安装 networkx 和 matplotlib 来可视化归因图")