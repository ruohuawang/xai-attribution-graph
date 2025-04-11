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
        attn_implementation="eager",  # 显式使用eager attention实现
        torch_dtype=torch.float16,    # 使用半精度加载模型，减少内存占用
        device_map="auto"             # 自动处理模型在设备间的分配
    )
    
    # 将模型移至指定设备
    if device != "auto":
        model = model.to(device)
    print(f"模型已加载完成")
    
    return model