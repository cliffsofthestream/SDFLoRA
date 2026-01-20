#!/usr/bin/env python3
"""
检查MNLI数据集的标签值范围
"""

import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def check_mnli_labels():
    """检查MNLI数据集的标签值"""
    print("检查MNLI数据集的标签值...")
    
    try:
        # 加载MNLI数据集
        datasets = load_dataset("glue", "mnli")
        
        # 检查标签
        train_labels = datasets["train"]["label"]
        val_labels = datasets["validation_matched"]["label"]
        
        print(f"训练集标签范围: {min(train_labels)} - {max(train_labels)}")
        print(f"验证集标签范围: {min(val_labels)} - {max(val_labels)}")
        print(f"训练集唯一标签: {sorted(set(train_labels))}")
        print(f"验证集唯一标签: {sorted(set(val_labels))}")
        
        # 检查标签名称
        label_names = datasets["train"].features["label"].names
        print(f"标签名称: {label_names}")
        print(f"标签数量: {len(label_names)}")
        
        # 检查是否有超出范围的标签
        num_labels = len(label_names)
        train_out_of_range = [l for l in train_labels if l >= num_labels or l < 0]
        val_out_of_range = [l for l in val_labels if l >= num_labels or l < 0]
        
        if train_out_of_range:
            print(f"❌ 训练集中有超出范围的标签: {train_out_of_range}")
        else:
            print("✅ 训练集标签值在有效范围内")
            
        if val_out_of_range:
            print(f"❌ 验证集中有超出范围的标签: {val_out_of_range}")
        else:
            print("✅ 验证集标签值在有效范围内")
        
        # 检查数据加载后的格式
        print("\n检查数据加载后的格式...")
        
        # 模拟数据加载过程
        model_name = "/home/szk_25/FederatedLLM/llama-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        def preprocess_function(examples):
            sentence1_key, sentence2_key = "premise", "hypothesis"
            args = (examples[sentence1_key], examples[sentence2_key])
            result = tokenizer(*args, padding='max_length', max_length=128, truncation=True)
            return result
        
        datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
        datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # 检查几个样本
        train_dataset = datasets["train"]
        for i in range(min(5, len(train_dataset))):
            sample = train_dataset[i]
            print(f"样本 {i}: 标签值 = {sample['label']}, 标签类型 = {type(sample['label'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== MNLI数据集标签检查 ===")
    
    success = check_mnli_labels()
    
    if success:
        print("\n✅ 检查完成")
    else:
        print("\n❌ 检查失败")
        sys.exit(1)

