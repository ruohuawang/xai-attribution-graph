from transformers import (
    AutoModelForCausalLM, 
    GPT2Model, 
    BertModel, 
    AutoTokenizer, 
    GPT2Tokenizer, 
    BertTokenizer
)
import os
from config import CONFIG, MODEL_PATHS

def load_model(model_path=None, model_type=None, device="cuda"):
    """
    加载预训练模型
    
    Args:
        model_path: 模型路径，如果为None则使用CONFIG中的路径
        model_type: 模型类型，如果为None则使用CONFIG中的类型
        device: 设备
        
    Returns:
        model: 加载的模型
    """
    if model_type is None:
        model_type = CONFIG["model_type"]
    
    if model_path is None:
        model_path = MODEL_PATHS.get(model_type, CONFIG["model_path"])
    
    # 转换为绝对路径
    if not os.path.isabs(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, model_path)
    
    print(f"加载模型: {model_type} 从路径: {model_path}")
    
    # 根据模型类型加载不同的模型
    if model_type == "qwen2":
        model = AutoModelForCausalLM.from_pretrained(model_path)
    elif model_type == "gpt2":
        model = GPT2Model.from_pretrained(model_path)
    elif model_type == "bert":
        model = BertModel.from_pretrained(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model.to(device)

def load_tokenizer(model_path=None, model_type=None):
    """
    加载tokenizer
    
    Args:
        model_path: 模型路径，如果为None则使用CONFIG中的路径
        model_type: 模型类型，如果为None则使用CONFIG中的类型
        
    Returns:
        tokenizer: 加载的tokenizer
    """
    if model_type is None:
        model_type = CONFIG["model_type"]
    
    if model_path is None:
        model_path = MODEL_PATHS.get(model_type, CONFIG["model_path"])
    
    # 转换为绝对路径
    if not os.path.isabs(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, model_path)
    
    # 根据模型类型加载不同的tokenizer
    if model_type == "qwen2":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
        # 确保GPT-2 tokenizer有pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return tokenizer