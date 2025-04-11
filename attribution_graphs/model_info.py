import argparse
from transformers import (
    AutoModelForCausalLM, 
    GPT2Model, 
    BertModel, 
    AutoTokenizer, 
    GPT2Tokenizer, 
    BertTokenizer
)
from config import CONFIG, MODEL_PATHS

def get_model_info(model_type=None):
    """获取模型信息并打印"""
    if model_type is None:
        model_type = CONFIG["model_type"]
    
    model_path = MODEL_PATHS.get(model_type, CONFIG["model_path"])
    print(f"加载模型: {model_type} 从路径: {model_path}")
    
    # 根据模型类型加载不同的模型
    if model_type == "qwen2":
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == "gpt2":
        model = GPT2Model.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    elif model_type == "bert":
        model = BertModel.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print('模型类型:', type(model).__name__)
    
    # 获取模型配置
    if hasattr(model, 'config'):
        config = model.config
        print('\n模型配置:')
        print(f'- hidden_size: {config.hidden_size}')
        
        # 不同模型的层数属性名可能不同
        if hasattr(config, 'num_hidden_layers'):
            print(f'- num_layers: {config.num_hidden_layers}')
        elif hasattr(config, 'n_layer'):
            print(f'- num_layers: {config.n_layer}')
        else:
            print('- num_layers: 未知')
        
        # 打印其他可能有用的配置
        if hasattr(config, 'num_attention_heads'):
            print(f'- num_attention_heads: {config.num_attention_heads}')
        if hasattr(config, 'vocab_size'):
            print(f'- vocab_size: {config.vocab_size}')
    
    # 查找模型的层结构
    print('\n模型层结构:')
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            print('- 层路径: model.model.layers')
            print(f'- 层数: {len(model.model.layers)}')
        elif hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layers'):
            print('- 层路径: model.model.encoder.layers')
            print(f'- 层数: {len(model.model.encoder.layers)}')
    elif hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            print('- 层路径: model.transformer.h')
            print(f'- 层数: {len(model.transformer.h)}')
        elif hasattr(model.transformer, 'layers'):
            print('- 层路径: model.transformer.layers')
            print(f'- 层数: {len(model.transformer.layers)}')
    elif hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'layers'):
            print('- 层路径: model.encoder.layers')
            print(f'- 层数: {len(model.encoder.layers)}')
    
    # 打印MLP路径信息
    print('\nMLP路径信息:')
    if model_type == "qwen2":
        print('- MLP路径: model.transformer.h[layer_idx].mlp')
    elif model_type == "gpt2":
        print('- MLP路径: model.h[layer_idx].mlp')
    elif model_type == "bert":
        print('- MLP路径: model.encoder.layer[layer_idx].intermediate')
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="获取模型信息")
    parser.add_argument("--model_type", type=str, choices=["qwen2", "gpt2", "bert"], 
                        default=CONFIG["model_type"], help="模型类型")
    args = parser.parse_args()
    
    get_model_info(args.model_type)