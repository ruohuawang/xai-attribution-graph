import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

class OpenWebTextDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, seq_length=128, cache_dir=None):
        self.seq_length = seq_length
        
        # 加载Qwen tokenizer
        print(f"加载Qwen tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 检查是否有缓存的分词结果
        self.cache_path = os.path.join(cache_dir, f"tokenized_openwebtext_{seq_length}.pt") if cache_dir else None
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"从缓存加载分词结果: {self.cache_path}")
            self.tokenized_texts = torch.load(self.cache_path)
            self.size = len(self.tokenized_texts)
            print(f"加载了{self.size}个分词后的样本")
            return
        
        # 直接从Hugging Face加载数据集
        print(f"从Hugging Face加载数据集: stas/openwebtext-10k")
        dataset = load_dataset("stas/openwebtext-10k", split='train')
        self.texts = [item['text'] for item in dataset]
        
        self.size = len(self.texts)
        print(f"加载了{self.size}个文本样本")
        
        # 预处理和分词
        print("开始对文本进行分词...")
        self.tokenized_texts = []
        
        # 使用批处理来提高效率
        batch_size = 32
        num_batches = (self.size + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, self.size, batch_size), desc="分词处理"):
            # 获取当前批次的文本
            batch_texts = self.texts[i:min(i+batch_size, self.size)]
            
            # 批量分词
            tokens = self.tokenizer(
                batch_texts, 
                truncation=True, 
                max_length=seq_length, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            # 将分词结果添加到列表中
            for j in range(len(batch_texts)):
                self.tokenized_texts.append(tokens.input_ids[j])
        
        self.size = len(self.tokenized_texts)
        print(f"处理完成，共{self.size}个分词后的样本")
        
        # 保存分词结果到缓存
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            print(f"保存分词结果到缓存: {self.cache_path}")
            torch.save(self.tokenized_texts, self.cache_path)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 获取已分词的文本
        x = self.tokenized_texts[idx]
            
        # 使用移位的序列作为目标（用于语言模型训练）
        y = torch.roll(x, -1)
        y[-1] = 0  # 最后一个位置填充
        return x, y

def get_dataloader(data_path, tokenizer_path, batch_size=16, seq_length=128):
    # 创建缓存目录
    cache_dir = os.path.join(os.path.dirname(data_path), "cache")
    
    dataset = OpenWebTextDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        seq_length=seq_length,
        cache_dir=cache_dir
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True  # 使用固定内存提高GPU传输速度
    )
    
    return dataloader