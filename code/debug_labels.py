#!/usr/bin/env python3
"""
调试标签分布和范围
"""

import sys
import os
sys.path.append('/home/szk_25/FedSA-LoRA')
sys.path.append('/home/szk_25/FedSA-LoRA-Dual')

from federatedscope.core.configs.config import CN
from federatedscope.glue.dataloader.dataloader import load_glue_dataset
from code.dual_lora_model_builder import get_dual_lora_llm
import torch

def debug_labels():
    print("=== 调试标签分布 ===")
    
    # 加载配置
    config = CN()
    config.defrost()
    
    # 数据配置
    config.data = CN()
    config.data.root = '/home/szk_25/FedSA-LoRA-Dual/GLUE'
    config.data.type = 'mnli@huggingface'
    config.data.matched = True
    config.data.label_list = []  # 预设置
    config.data.num_labels = 0  # 预设置
    
    # 模型配置
    config.model = CN()
    config.model.type = 'llama@huggingface'
    
    # LLM配置
    config.llm = CN()
    config.llm.cache = CN()
    config.llm.cache.model = '/home/szk_25/FederatedLLM'
    config.llm.tok_len = 512
    
    # 添加必要的配置
    config.outdir = '/tmp/debug'
    config.wandb = CN()
    config.wandb.use = False
    
    # 加载数据集
    print("加载数据集...")
    dataset, config = load_glue_dataset(config)
    train_dataset, eval_dataset, test_dataset = dataset
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    print(f"配置的num_labels: {config.data.num_labels}")
    print(f"标签列表: {config.data.label_list}")
    
    # 检查训练集标签分布
    print("\n=== 训练集标签分布 ===")
    train_labels = train_dataset['label']
    print(f"训练集标签类型: {type(train_labels)}")
    print(f"训练集标签形状: {train_labels.shape if hasattr(train_labels, 'shape') else 'N/A'}")
    print(f"训练集标签范围: {train_labels.min().item()} - {train_labels.max().item()}")
    print(f"训练集唯一标签: {torch.unique(train_labels)}")
    
    # 检查验证集标签分布
    print("\n=== 验证集标签分布 ===")
    eval_labels = eval_dataset['label']
    print(f"验证集标签类型: {type(eval_labels)}")
    print(f"验证集标签形状: {eval_labels.shape if hasattr(eval_labels, 'shape') else 'N/A'}")
    print(f"验证集标签范围: {eval_labels.min().item()} - {eval_labels.max().item()}")
    print(f"验证集唯一标签: {torch.unique(eval_labels)}")
    
    # 检查前几个样本
    print("\n=== 前5个训练样本标签 ===")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"样本 {i}: label={sample['label']}")
    
    print("\n=== 前5个验证样本标签 ===")
    for i in range(min(5, len(eval_dataset))):
        sample = eval_dataset[i]
        print(f"样本 {i}: label={sample['label']}")
    
    # 创建模型并检查
    print("\n=== 模型检查 ===")
    model = get_dual_lora_llm(config)
    print(f"模型类型: {type(model)}")
    
    # 检查分类器层
    if hasattr(model, 'score'):
        print(f"分类器层权重形状: {model.score.weight.shape}")
        print(f"分类器层偏置形状: {model.score.bias.shape if hasattr(model.score, 'bias') else 'No bias'}")
    else:
        print("未找到分类器层")
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    sample = train_dataset[0]
    input_ids = sample['input_ids'].unsqueeze(0)  # 添加batch维度
    attention_mask = sample['attention_mask'].unsqueeze(0)
    label = sample['label'].unsqueeze(0)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"标签值: {label.item()}")
    print(f"标签是否在范围内: {0 <= label.item() < config.data.num_labels}")
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          labels=label)
        print(f"✅ 前向传播成功")
        print(f"输出logits形状: {outputs.logits.shape}")
        print(f"损失值: {outputs.loss.item()}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")

if __name__ == "__main__":
    debug_labels()