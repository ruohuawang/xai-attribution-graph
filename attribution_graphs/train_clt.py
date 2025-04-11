import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime

from config import CLT_CONFIG, CONFIG
from model_loader import load_model
from data_loader import get_dataloader
from models.clt import CrossLayerTranscoder
import argparse

def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_clt(model, dataloader, clt_config, output_dir):
    """
    训练Cross-Layer Transcoder (CLT)
    
    Args:
        model: 预训练的语言模型
        dataloader: 数据加载器
        clt_config: CLT配置
        output_dir: 输出目录
    """
    device = torch.device(clt_config["device"])
    model_type = CONFIG.get("model_type", "qwen2")
    
    # 获取模型的实际隐藏层大小，用于输入适配
    # 获取模型的隐藏层大小和层数
    if hasattr(model, 'config'):
        actual_hidden_size = model.config.hidden_size
        
        # 根据不同模型类型获取层数
        if hasattr(model.config, 'num_hidden_layers'):
            actual_num_layers = model.config.num_hidden_layers
        elif hasattr(model.config, 'n_layer'):
            actual_num_layers = model.config.n_layer
        else:
            # 尝试从模型结构推断层数
            actual_num_layers = clt_config["num_layers"]
            print(f"警告: 无法从模型配置获取层数，使用配置中的值: {actual_num_layers}")
        
        # 保存实际隐藏层大小，但不更新配置的hidden_size和num_layers
        clt_config["actual_hidden_size"] = actual_hidden_size
        print(f"模型实际配置: hidden_size={actual_hidden_size}, num_layers={actual_num_layers}")
        print(f"CLT使用配置: hidden_size={clt_config['hidden_size']}, num_layers={clt_config['num_layers']}")
    else:
        actual_hidden_size = clt_config["hidden_size"]
        actual_num_layers = clt_config["num_layers"]
    
    # 保存实际隐藏层大小，用于后续处理
    clt_config["actual_hidden_size"] = actual_hidden_size
    
    print(f"CLT配置: {clt_config}")
    
    # 初始化CLT模型
    clt = CrossLayerTranscoder(clt_config).to(device)
    
    # 如果实际隐藏层大小与配置不同，添加维度适配层
    input_adapters = []
    if actual_hidden_size != clt_config["hidden_size"]:
        for i in range(clt_config["num_layers"]):
            adapter = nn.Linear(actual_hidden_size, clt_config["hidden_size"]).to(device)
            input_adapters.append(adapter)
    
    # 创建优化器 - 添加适配层参数
    optimizer_params = list(clt.parameters())
    for adapter in input_adapters:
        optimizer_params.extend(adapter.parameters())
    
    optimizer = optim.AdamW(
        optimizer_params,
        lr=clt_config["learning_rate"],
        weight_decay=clt_config["weight_decay"]
    )
    
    # 创建学习率调度器
    # 修复KeyError: 'max_batches_per_epoch'
    total_steps = len(dataloader) * clt_config["epochs"]
    if "max_batches_per_epoch" in clt_config:
        total_steps = min(total_steps, clt_config["max_batches_per_epoch"] * clt_config["epochs"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, int(total_steps * 0.1)),
        num_training_steps=total_steps
    )
    
    # 混合精度训练 - 更新为新API
    scaler = torch.amp.GradScaler('cuda') if clt_config.get("fp16", False) else None
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 训练循环
    global_step = 0
    for epoch in range(clt_config["epochs"]):
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_sparsity_loss = 0.0
        
        # 创建进度条 - 修复KeyError
        max_batches = len(dataloader)
        if "max_batches_per_epoch" in clt_config:
            max_batches = min(max_batches, clt_config["max_batches_per_epoch"])
        
        progress_bar = tqdm(total=max_batches, desc=f"Epoch {epoch+1}/{clt_config['epochs']}")
        
        # 将模型设置为评估模式，我们只需要前向传播来获取激活值
        model.eval()
        # 将CLT设置为训练模式
        clt.train()
        
        for batch_idx, (input_ids, _) in enumerate(dataloader):
            # 检查是否达到每个epoch的最大批次数 - 修复KeyError
            if "max_batches_per_epoch" in clt_config and batch_idx >= clt_config["max_batches_per_epoch"]:
                break
                
            # 将输入移至设备
            input_ids = input_ids.to(device)
            
            # 梯度累积步数
            accumulation_steps = clt_config.get("gradient_accumulation_steps", 1)
            
            # 捕获模型的残差流激活值和MLP输出
            with torch.no_grad():
                # 存储残差流激活值和MLP输出
                residual_stream_activations = []
                mlp_outputs = []
                
                # 注册钩子以捕获残差流激活值和MLP输出
                hooks = []
                
                def capture_residual_stream(module, input, output, layer_idx):
                    # 确保列表长度足够
                    while len(residual_stream_activations) <= layer_idx:
                        residual_stream_activations.append(None)
                    # 处理输出可能是元组的情况
                    if isinstance(output, tuple):
                        residual_stream_activations[layer_idx] = output[0]  # 通常第一个元素是主要的隐藏状态
                    else:
                        residual_stream_activations[layer_idx] = output
                
                def capture_mlp_output(module, input, output, layer_idx):
                    # 确保列表长度足够
                    while len(mlp_outputs) <= layer_idx:
                        mlp_outputs.append(None)
                    # 处理输出可能是元组的情况
                    if isinstance(output, tuple):
                        mlp_outputs[layer_idx] = output[0]  # 通常第一个元素是主要的输出
                    else:
                        mlp_outputs[layer_idx] = output
                
                # 根据不同模型类型注册钩子
                if model_type == "qwen2":
                    # Qwen模型结构
                    for i, layer in enumerate(model.transformer.h):
                        # 捕获残差流激活值（在MLP之前）
                        hooks.append(layer.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                        ))
                        
                        # 捕获MLP输出
                        hooks.append(layer.mlp.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                        ))
                
                elif model_type == "gpt2":
                    # GPT-2模型结构
                    for i, layer in enumerate(model.h):
                        # 捕获残差流激活值
                        hooks.append(layer.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                        ))
                        
                        # 捕获MLP输出
                        hooks.append(layer.mlp.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                        ))
                
                elif model_type == "bert":
                    # BERT模型结构
                    for i, layer in enumerate(model.encoder.layer):
                        # 捕获残差流激活值
                        hooks.append(layer.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                        ))
                        
                        # 捕获MLP输出 (BERT中称为intermediate)
                        hooks.append(layer.intermediate.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                        ))
                
                else:
                    # 通用方法，尝试自动检测模型结构
                    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                        layers = model.transformer.h
                        mlp_name = 'mlp'
                    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                        layers = model.model.layers
                        mlp_name = 'feed_forward'
                    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                        layers = model.encoder.layer
                        mlp_name = 'intermediate'
                    else:
                        raise ValueError(f"不支持的模型结构: {type(model).__name__}")
                    
                    for i, layer in enumerate(layers):
                        # 捕获残差流激活值
                        hooks.append(layer.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                        ))
                        
                        # 捕获MLP输出
                        mlp = getattr(layer, mlp_name, None)
                        if mlp is not None:
                            hooks.append(mlp.register_forward_hook(
                                lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                            ))
                
                # 前向传播 - 根据模型类型调整输入格式
                if model_type == "bert":
                    # BERT需要attention_mask
                    attention_mask = torch.ones_like(input_ids)
                    outputs = model(input_ids, attention_mask=attention_mask)
                else:
                    # GPT-2和Qwen可以只使用input_ids
                    outputs = model(input_ids)
                
                # 移除钩子
                for hook in hooks:
                    hook.remove()
            
            # 使用混合精度训练 - 更新为新API
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # 如果需要，应用维度适配
                if actual_hidden_size != clt_config["hidden_size"] and input_adapters:
                    adapted_activations = []
                    for i, activation in enumerate(residual_stream_activations):
                        if i < len(input_adapters):
                            adapted_activations.append(input_adapters[i](activation))
                        else:
                            # 如果层数超出预期，截断
                            break
                    
                    # 使用CLT重建MLP输出
                    reconstructed_outputs, feature_activations = clt(adapted_activations)
                else:
                    # 使用原始激活值
                    reconstructed_outputs, feature_activations = clt(residual_stream_activations)
                
                # 计算损失
                loss, mse_loss, sparsity_loss = clt.compute_loss(
                    reconstructed_outputs, mlp_outputs, feature_activations
                )
                
                # 缩放损失以适应梯度累积
                loss = loss / accumulation_steps
            
            # 反向传播
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                # 梯度裁剪
                if scaler:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(clt.parameters(), clt_config["max_grad_norm"])
                
                # 更新参数
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                # 清零梯度
                optimizer.zero_grad()
            
            # 更新进度条
            progress_bar.update(1)
            
            # 更新损失
            epoch_loss += loss.item() * accumulation_steps
            epoch_mse_loss += mse_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            
            # 打印当前批次的损失
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item() * accumulation_steps:.4f}, "
                      f"MSE Loss: {mse_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
            
            global_step += 1
        
        # 计算平均损失 - 修复KeyError
        max_batches = min(len(dataloader), batch_idx + 1)
        if "max_batches_per_epoch" in clt_config:
            max_batches = min(max_batches, clt_config["max_batches_per_epoch"])
        
        avg_loss = epoch_loss / max_batches
        avg_mse_loss = epoch_mse_loss / max_batches
        avg_sparsity_loss = epoch_sparsity_loss / max_batches
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{clt_config['epochs']}, "
              f"Avg Loss: {avg_loss:.4f}, "
              f"Avg MSE Loss: {avg_mse_loss:.4f}, "
              f"Avg Sparsity Loss: {avg_sparsity_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % clt_config["save_interval"] == 0:
            save_path = os.path.join(output_dir, f"clt_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'clt_state_dict': clt.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': avg_loss,
                'config': clt_config,
                'model_type': model_type  # 保存模型类型信息
            }, save_path)
            print(f"模型已保存到 {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(output_dir, "clt_final.pt")
    torch.save({
        'epoch': clt_config["epochs"],
        'clt_state_dict': clt.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': avg_loss,
        'config': clt_config,
        'model_type': model_type  # 保存模型类型信息
    }, final_save_path)
    print(f"最终模型已保存到 {final_save_path}")
    
    return clt

def evaluate_clt(model, clt, dataloader, device):
    """
    评估CLT模型性能
    
    Args:
        model: 预训练的语言模型
        clt: CLT模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        avg_loss: 平均损失
    """
    model.eval()
    clt.eval()
    
    total_loss = 0.0
    total_mse_loss = 0.0
    total_sparsity_loss = 0.0
    
    model_type = CONFIG.get("model_type", "qwen2")
    
    with torch.no_grad():
        for batch_idx, (input_ids, _) in enumerate(tqdm(dataloader, desc="评估CLT")):
            # 将输入移至设备
            input_ids = input_ids.to(device)
            
            # 捕获模型的残差流激活值和MLP输出
            residual_stream_activations = []
            mlp_outputs = []
            
            # 注册钩子以捕获残差流激活值和MLP输出
            hooks = []
            
            def capture_residual_stream(module, input, output, layer_idx):
                # 确保列表长度足够
                while len(residual_stream_activations) <= layer_idx:
                    residual_stream_activations.append(None)
                # 处理输出可能是元组的情况
                if isinstance(output, tuple):
                    residual_stream_activations[layer_idx] = output[0]
                else:
                    residual_stream_activations[layer_idx] = output
            
            def capture_mlp_output(module, input, output, layer_idx):
                # 确保列表长度足够
                while len(mlp_outputs) <= layer_idx:
                    mlp_outputs.append(None)
                # 处理输出可能是元组的情况
                if isinstance(output, tuple):
                    mlp_outputs[layer_idx] = output[0]
                else:
                    mlp_outputs[layer_idx] = output
            
            # 根据不同模型类型注册钩子
            if model_type == "qwen2":
                # Qwen模型结构
                for i, layer in enumerate(model.transformer.h):
                    hooks.append(layer.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                    ))
                    hooks.append(layer.mlp.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                    ))
            elif model_type == "gpt2":
                # GPT-2模型结构
                for i, layer in enumerate(model.h):
                    hooks.append(layer.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                    ))
                    hooks.append(layer.mlp.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                    ))
            elif model_type == "bert":
                # BERT模型结构
                for i, layer in enumerate(model.encoder.layer):
                    hooks.append(layer.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                    ))
                    hooks.append(layer.intermediate.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                    ))
            else:
                # 通用方法，尝试自动检测模型结构
                if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    layers = model.transformer.h
                    mlp_name = 'mlp'
                elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    layers = model.model.layers
                    mlp_name = 'feed_forward'
                elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
                    layers = model.encoder.layer
                    mlp_name = 'intermediate'
                else:
                    raise ValueError(f"不支持的模型结构: {type(model).__name__}")
                
                for i, layer in enumerate(layers):
                    hooks.append(layer.register_forward_hook(
                        lambda module, input, output, layer_idx=i: capture_residual_stream(module, input, output, layer_idx)
                    ))
                    
                    mlp = getattr(layer, mlp_name, None)
                    if mlp is not None:
                        hooks.append(mlp.register_forward_hook(
                            lambda module, input, output, layer_idx=i: capture_mlp_output(module, input, output, layer_idx)
                        ))
            
            # 前向传播 - 根据模型类型调整输入格式
            if model_type == "bert":
                # BERT需要attention_mask
                attention_mask = torch.ones_like(input_ids)
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                # GPT-2和Qwen可以只使用input_ids
                outputs = model(input_ids)
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 使用CLT重建MLP输出
            reconstructed_outputs, feature_activations = clt(residual_stream_activations)
            
            # 计算损失
            loss, mse_loss, sparsity_loss = clt.compute_loss(
                reconstructed_outputs, mlp_outputs, feature_activations
            )
            
            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_sparsity_loss += sparsity_loss.item()
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_mse_loss = total_mse_loss / num_batches
    avg_sparsity_loss = total_sparsity_loss / num_batches
    
    print(f"评估结果 - Avg Loss: {avg_loss:.4f}, "
          f"Avg MSE Loss: {avg_mse_loss:.4f}, "
          f"Avg Sparsity Loss: {avg_sparsity_loss:.4f}")
    
    return avg_loss

def load_clt(clt_path, device="cuda"):
    """加载CLT模型"""
    checkpoint = torch.load(clt_path, map_location=device)
    clt_config = checkpoint["config"]
    model_type = checkpoint.get("model_type", "qwen2")  # 获取保存的模型类型，默认为qwen2
    
    clt = CrossLayerTranscoder(clt_config).to(device)
    clt.load_state_dict(checkpoint["clt_state_dict"])
    clt.eval()
    
    return clt, clt_config, model_type

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练和评估Cross-Layer Transcoder (CLT)")
    parser.add_argument("--train", action="store_true", help="训练CLT模型")
    parser.add_argument("--eval", action="store_true", help="评估CLT模型")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--clt_path", type=str, default=None, help="CLT模型路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认执行训练
    if not (args.train or args.eval):
        args.train = True
        print("未指定操作，默认执行训练...")
    
    # 设置随机种子
    set_seed(CLT_CONFIG["seed"])
    
    # 处理命令行参数
    if args.model_path:
        CONFIG["model_path"] = args.model_path
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 创建默认输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(CONFIG["clt_dir"], f"clt_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.train:
        # 加载预训练模型
        model = load_model(CONFIG["model_path"], CONFIG.get("model_type"), device=CLT_CONFIG["device"])
        
        # 获取数据加载器
        dataloader = get_dataloader(
            data_path=CONFIG["data_path"],
            model_type=CONFIG.get("model_type"),
            model_path=CONFIG["model_path"],
            batch_size=CLT_CONFIG["batch_size"],
            seq_length=CONFIG["seq_length"]
        )
        
        # 训练CLT
        clt = train_clt(model, dataloader, CLT_CONFIG, output_dir)
        print("CLT训练完成！")
    
    if args.eval:
        if not args.clt_path:
            print("错误: 评估模式需要指定CLT模型路径 (--clt_path)")
            exit(1)
        
        # 加载CLT模型
        clt, clt_config, model_type = load_clt(args.clt_path, device=CLT_CONFIG["device"])
        
        # 如果模型类型与配置不同，更新配置
        if model_type != CONFIG.get("model_type"):
            print(f"警告: CLT模型的模型类型 ({model_type}) 与配置中的类型 ({CONFIG.get('model_type')}) 不同")
            print(f"使用CLT模型的模型类型: {model_type}")
            CONFIG["model_type"] = model_type
        
        # 加载预训练模型
        model = load_model(CONFIG["model_path"], CONFIG.get("model_type"), device=CLT_CONFIG["device"])
        
        # 获取数据加载器
        dataloader = get_dataloader(
            data_path=CONFIG["data_path"],
            model_type=CONFIG.get("model_type"),
            model_path=CONFIG["model_path"],
            batch_size=CLT_CONFIG["batch_size"],
            seq_length=CONFIG["seq_length"]
        )
        
        # 评估CLT
        evaluate_clt(model, clt, dataloader, CLT_CONFIG["device"])
        print("CLT评估完成！")