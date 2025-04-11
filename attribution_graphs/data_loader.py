import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from config import CONFIG, MODEL_PATHS

class OpenWebTextDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, model_type=None, model_path=None, seq_length=128, cache_dir=None):
        self.seq_length = seq_length
        
        # 根据model_type选择合适的tokenizer
        if tokenizer is None:
            if model_type is None:
                model_type = CONFIG["model_type"]
            
            if model_path is None:
                model_path = MODEL_PATHS.get(model_type, CONFIG["model_path"])
            
            print(f"加载{model_type} tokenizer: {model_path}")
            
            # 根据模型类型加载不同的tokenizer
            if model_type == "qwen2":
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            elif model_type == "gpt2":
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
                # 确保GPT-2 tokenizer有pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            elif model_type == "bert":
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
            else:
                # 默认使用AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        
        # 检查是否有缓存的分词结果
        cache_name = f"tokenized_openwebtext_{model_type}_{seq_length}.pt"
        self.cache_path = os.path.join(cache_dir, cache_name) if cache_dir else None
        
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

def get_dataloader(data_path=None, model_type=None, model_path=None, batch_size=None, seq_length=None):
    """
    获取数据加载器
    
    Args:
        data_path: 数据路径，如果为None则使用CONFIG中的路径
        model_type: 模型类型，如果为None则使用CONFIG中的类型
        model_path: 模型路径，如果为None则根据model_type从MODEL_PATHS获取
        batch_size: 批量大小，如果为None则使用CONFIG中的值
        seq_length: 序列长度，如果为None则使用CONFIG中的值
        
    Returns:
        dataloader: 数据加载器
    """
    # 使用CONFIG中的默认值
    if data_path is None:
        data_path = CONFIG["data_path"]
    if model_type is None:
        model_type = CONFIG["model_type"]
    if model_path is None:
        model_path = MODEL_PATHS.get(model_type, CONFIG["model_path"])
    if batch_size is None:
        batch_size = CONFIG["batch_size"]
    if seq_length is None:
        seq_length = CONFIG["seq_length"]
    
    # 创建缓存目录
    cache_dir = os.path.join(os.path.dirname(data_path), "cache")
    
    dataset = OpenWebTextDataset(
        data_path=data_path,
        model_type=model_type,
        model_path=model_path,
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