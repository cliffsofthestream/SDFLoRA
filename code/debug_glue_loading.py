#!/usr/bin/env python3
"""
调试GLUE数据加载过程中的标签处理
"""

import os
import sys
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

def test_glue_data_loading():
    """测试GLUE数据加载过程"""
    print("测试GLUE数据加载过程...")
    
    # 模拟配置
    class Config:
        def __init__(self):
            self.data = type('obj', (object,), {})()
            self.data.root = "/home/user/FedSA-LoRA-Dual/GLUE"
            self.data.type = "mnli@glue"
            self.data.matched = True
            self.data.num_labels = 3
            self.llm = type('obj', (object,), {})()
            self.llm.tok_len = 128
            self.llm.cache = type('obj', (object,), {})()
            self.llm.cache.model = "/home/user/FederatedLLM/llama-7b"
            self.model = type('obj', (object,), {})()
            self.model.type = "/home/user/FederatedLLM/llama-7b@huggingface_llm"
    
    config = Config()
    
    try:
        # 加载数据集
        task_name = "mnli"
        model_name = "/home/user/FederatedLLM/llama-7b"
        
        # 尝试从本地缓存加载
        cache_path = os.path.join(config.data.root, task_name)
        print(f"缓存路径: {cache_path}")
        
        if os.path.exists(cache_path):
            print("✅ 找到本地缓存")
            # 直接使用parquet文件
            train_file = os.path.join(cache_path, "train-00000-of-00001.parquet")
            val_file = os.path.join(cache_path, "validation_matched-00000-of-00001.parquet")
            
            if os.path.exists(train_file) and os.path.exists(val_file):
                print("✅ 找到parquet文件")
                
                # 加载数据
                train_df = pd.read_parquet(train_file)
                val_df = pd.read_parquet(val_file)
                
                print(f"训练数据形状: {train_df.shape}")
                print(f"验证数据形状: {val_df.shape}")
                
                # 检查标签
                train_labels = train_df['label'].values
                val_labels = val_df['label'].values
                
                print(f"训练标签范围: {min(train_labels)} - {max(train_labels)}")
                print(f"验证标签范围: {min(val_labels)} - {max(val_labels)}")
                
                # 转换为Dataset对象
                from datasets import Dataset
                train_dataset = Dataset.from_pandas(train_df)
                val_dataset = Dataset.from_pandas(val_df)
                
                # 加载分词器
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # 预处理函数
                def preprocess_function(examples):
                    # MNLI使用premise和hypothesis
                    args = (examples['premise'], examples['hypothesis'])
                    result = tokenizer(*args, padding='max_length', max_length=128, truncation=True)
                    return result
                
                # 应用预处理
                train_dataset = train_dataset.map(preprocess_function, batched=True)
                val_dataset = val_dataset.map(preprocess_function, batched=True)
                
                # 设置格式
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
                val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
                
                # 检查处理后的标签
                print("\n处理后的标签检查:")
                train_labels_processed = train_dataset['label'][:10]  # 取前10个样本
                val_labels_processed = val_dataset['label'][:10]
                
                print(f"训练标签类型: {type(train_labels_processed)}")
                print(f"训练标签形状: {train_labels_processed.shape}")
                print(f"训练标签值: {train_labels_processed}")
                print(f"训练标签范围: {train_labels_processed.min()} - {train_labels_processed.max()}")
                
                print(f"验证标签类型: {type(val_labels_processed)}")
                print(f"验证标签形状: {val_labels_processed.shape}")
                print(f"验证标签值: {val_labels_processed}")
                print(f"验证标签范围: {val_labels_processed.min()} - {val_labels_processed.max()}")
                
                # 检查是否有超出范围的标签
                if train_labels_processed.max() >= 3 or train_labels_processed.min() < 0:
                    print("❌ 训练集标签超出范围!")
                    return False
                if val_labels_processed.max() >= 3 or val_labels_processed.min() < 0:
                    print("❌ 验证集标签超出范围!")
                    return False
                
                print("✅ 标签处理正常")
                return True
            else:
                print("❌ 未找到parquet文件")
                return False
        else:
            print("❌ 缓存路径不存在")
            return False
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GLUE数据加载调试 ===")
    
    success = test_glue_data_loading()
    
    if success:
        print("\n✅ 数据加载测试通过")
    else:
        print("\n❌ 数据加载测试失败")
        sys.exit(1)

