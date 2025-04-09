import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk, Dataset as HFDataset
from transformers import AutoTokenizer
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class OpenWebTextDataset(Dataset):
    def __init__(self, data_path, tokenizer_path, seq_length=128):
        self.seq_length = seq_length
        
        # 加载Qwen tokenizer
        print(f"加载Qwen tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        
        # 检查数据路径
        print(f"检查数据路径: {data_path}")
        
        # 尝试不同的方法加载数据
        self.texts = []
        
        # 方法1: 尝试直接加载arrow文件
        arrow_file = os.path.join(data_path, "openwebtext-10k-train.arrow")
        if os.path.exists(arrow_file):
            print(f"找到Arrow文件: {arrow_file}")
            try:
                # 使用pyarrow直接读取
                reader = pa.ipc.RecordBatchFileReader(arrow_file)
                batches = [batch for batch in reader]
                table = pa.Table.from_batches(batches)
                df = table.to_pandas()
                
                if 'text' in df.columns:
                    self.texts = df['text'].tolist()
                    print(f"从Arrow文件加载了{len(self.texts)}个文本样本")
                else:
                    print(f"Arrow文件中没有'text'列，列名为: {df.columns.tolist()}")
            except Exception as e:
                print(f"使用pyarrow加载Arrow文件失败: {e}")
                
                # 尝试修复损坏的Arrow文件
                try:
                    print("尝试修复损坏的Arrow文件...")
                    with open(arrow_file, 'rb') as f:
                        data = f.read()
                    
                    # 检查文件头是否是有效的Arrow格式
                    if not data.startswith(b'ARROW1'):
                        print("文件头无效，尝试从原始文本文件重新生成")
                        
                        # 查找同目录下的文本文件
                        txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
                        if txt_files:
                            print(f"找到{len(txt_files)}个文本文件，尝试重新生成Arrow文件...")
                            df = pd.DataFrame(columns=['text'])
                            
                            for txt_file in txt_files:
                                with open(os.path.join(data_path, txt_file), 'r', encoding='utf-8') as f:
                                    lines = [line.strip() for line in f if line.strip()]
                                    df = pd.concat([df, pd.DataFrame({'text': lines})], ignore_index=True)
                            
                            # 重新生成Arrow文件
                            table = pa.Table.from_pandas(df)
                            with pa.OSFile(arrow_file, 'wb') as sink:
                                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                                    writer.write_table(table)
                            print(f"成功重新生成Arrow文件: {arrow_file}")
                            
                            # 重新尝试加载
                            reader = pa.ipc.RecordBatchFileReader(arrow_file)
                            batches = [batch for batch in reader]
                            table = pa.Table.from_batches(batches)
                            df = table.to_pandas()
                            
                            if 'text' in df.columns:
                                self.texts = df['text'].tolist()
                                print(f"从修复后的Arrow文件加载了{len(self.texts)}个文本样本")
                                
                except Exception as e:
                    print(f"修复Arrow文件失败: {e}")
                
        # 方法1.5: 尝试加载本地文本文件
        txt_files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        if txt_files and not self.texts:
            print(f"找到{len(txt_files)}个文本文件，尝试加载...")
            try:
                for txt_file in txt_files:
                    with open(os.path.join(data_path, txt_file), 'r', encoding='utf-8') as f:
                        self.texts.extend([line.strip() for line in f if line.strip()])
                print(f"从文本文件加载了{len(self.texts)}个文本样本")
            except Exception as e:
                print(f"加载文本文件失败: {e}")
        
        # 方法2: 尝试加载数据集目录
        if not self.texts and os.path.isdir(data_path):
            print(f"尝试作为数据集目录加载: {data_path}")
            try:
                # 尝试从父目录加载
                parent_dir = os.path.dirname(data_path)
                dataset = load_from_disk(parent_dir)
                print(f"从父目录加载数据集成功: {parent_dir}")
                
                if isinstance(dataset, dict) and 'train' in dataset:
                    self.texts = [item['text'] for item in dataset['train']]
                else:
                    self.texts = [item['text'] for item in dataset]
                
                print(f"从数据集加载了{len(self.texts)}个文本样本")
            except Exception as e:
                print(f"从父目录加载数据集失败: {e}")
        
        # 方法3: 尝试直接从Hugging Face加载
        if not self.texts:
            print("尝试直接从Hugging Face加载数据集")
            try:
                dataset = load_dataset("stas/openwebtext-10k", split='train')
                self.texts = [item['text'] for item in dataset]
                print(f"从Hugging Face加载了{len(self.texts)}个文本样本")
            except Exception as e:
                print(f"从Hugging Face加载数据集失败: {e}")
        
        # 如果所有方法都失败，报错中断
        if not self.texts:
            raise RuntimeError("所有数据加载方法都失败，无法继续。请检查数据路径或网络连接。")
        
        self.size = len(self.texts)
        print(f"最终加载了{self.size}个文本样本")
        
        # 预处理和分词
        print("开始对文本进行分词...")
        self.tokenized_texts = []
        
        # 为了提高效率，只处理前1000个样本
        max_samples = min(1000, self.size)
        print(f"处理前{max_samples}个样本...")
        
        for i in range(max_samples):
            text = self.texts[i]
            tokens = self.tokenizer(
                text, 
                truncation=True, 
                max_length=seq_length, 
                padding="max_length", 
                return_tensors="pt"
            )
            self.tokenized_texts.append(tokens.input_ids[0])
        
        self.size = len(self.tokenized_texts)
        print(f"处理完成，共{self.size}个分词后的样本")
    
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
    dataset = OpenWebTextDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        seq_length=seq_length
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # 设为0避免多进程问题
    )
    
    return dataloader