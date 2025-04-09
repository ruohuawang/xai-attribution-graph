import torch
from transformers import AutoModelForCausalLM, AutoConfig

def load_model(model_path, device="cuda"):
    print(f"从{model_path}加载Qwen模型...")
    
    # 加载模型配置
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # 修改配置以适应归因图分析
    config.output_hidden_states = True
    config.output_attentions = True
    config.use_cache = False
    
    # 加载模型，显式指定使用eager attention实现
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        attn_implementation="eager"  # 显式使用eager attention实现
    )
    
    # 将模型移至指定设备
    model = model.to(device)
    print(f"模型已加载到{device}设备")
    
    return model