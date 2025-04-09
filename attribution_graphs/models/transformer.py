import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size = x.size(0)
        
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头形式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        output = self.out_proj(context)
        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # 自注意力层
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 前馈层
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1024, config["hidden_size"]))
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                config["hidden_size"], 
                config["num_heads"], 
                config["dropout"]
            ) for _ in range(config["num_layers"])
        ])
        
        self.norm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
        
        # 输出层
        self.output = nn.Linear(config["hidden_size"], config["vocab_size"])
        
    def forward(self, x):
        seq_length = x.size(1)
        
        # 词嵌入和位置嵌入
        token_embeddings = self.embedding(x)
        position_embeddings = self.pos_embedding[:, :seq_length, :]
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # 输出层
        logits = self.output(x)
        
        return logits