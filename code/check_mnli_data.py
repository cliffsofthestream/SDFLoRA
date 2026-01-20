

import os
import sys
import torch
import pandas as pd
from datasets import Dataset

def check_local_mnli_data():
    """检查本地MNLI数据"""
    print("检查本地MNLI数据...")
    
    glue_path = "/home/user/FedSA-LoRA-Dual/GLUE"
    mnli_path = os.path.join(glue_path, "mnli")
    
    if not os.path.exists(mnli_path):
        print(f"❌ MNLI路径不存在: {mnli_path}")
        return False
    
    # 检查训练数据
    train_file = os.path.join(mnli_path, "train-00000-of-00001.parquet")
    if os.path.exists(train_file):
        print(f"✅ 找到训练文件: {train_file}")
        try:
            df = pd.read_parquet(train_file)
            print(f"训练数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            if 'label' in df.columns:
                labels = df['label'].values
                print(f"标签唯一值: {set(labels)}")
                print(f"标签范围: {min(labels)} - {max(labels)}")
                print(f"标签数据类型: {type(labels[0])}")
                
                # 检查是否有超出范围的标签
                if max(labels) >= 3 or min(labels) < 0:
                    print("❌ 发现超出范围的标签!")
                    return False
                else:
                    print("✅ 标签在正确范围内 [0, 2]")
            else:
                print("❌ 未找到'label'列")
                return False
                
        except Exception as e:
            print(f"❌ 读取训练文件失败: {e}")
            return False
    else:
        print(f"❌ 训练文件不存在: {train_file}")
        return False
    
    # 检查验证数据
    val_file = os.path.join(mnli_path, "validation_matched-00000-of-00001.parquet")
    if os.path.exists(val_file):
        print(f"✅ 找到验证文件: {val_file}")
        try:
            df = pd.read_parquet(val_file)
            print(f"验证数据形状: {df.shape}")
            
            if 'label' in df.columns:
                labels = df['label'].values
                print(f"验证标签唯一值: {set(labels)}")
                print(f"验证标签范围: {min(labels)} - {max(labels)}")
                
                if max(labels) >= 3 or min(labels) < 0:
                    print("❌ 验证集发现超出范围的标签!")
                    return False
                else:
                    print("✅ 验证集标签在正确范围内 [0, 2]")
        except Exception as e:
            print(f"❌ 读取验证文件失败: {e}")
            return False
    
    return True

def test_model_forward():
    """测试模型前向传播"""
    print("\n测试模型前向传播...")
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # 加载模型和分词器
        model_path = "/home/user/FederatedLLM/llama-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=3,
            torch_dtype=torch.float16
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 创建测试数据
        test_texts = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Birds fly in the sky."
        ]
        
        # 分词
        inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 创建测试标签 (0, 1, 2)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            print(f"输出logits形状: {outputs.logits.shape}")
            print(f"损失值: {outputs.loss}")
            print("✅ 模型前向传播成功")
            
    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=== MNLI数据检查脚本 ===")
    
    # 检查本地数据
    data_ok = check_local_mnli_data()
    
    # 测试模型
    model_ok = test_model_forward()
    
    if data_ok and model_ok:
        print("\n✅ 所有检查通过")
    else:
        print("\n❌ 发现问题")
        sys.exit(1)
