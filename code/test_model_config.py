import os
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def test_model_config():
    """测试模型配置"""
    print("测试模型配置...")
    
    model_path = "/home/szk_25/FederatedLLM/llama-7b"
    
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试不同的num_labels配置
        for num_labels in [1, 2, 3]:
            print(f"\n测试 num_labels = {num_labels}")
            
            # 加载模型
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, 
                num_labels=num_labels,
                torch_dtype=torch.float16
            )
            
            print(f"模型分类器输出维度: {model.classifier.out_features}")
            print(f"模型num_labels: {model.num_labels}")
            
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
            
            # 创建测试标签
            if num_labels == 1:
                labels = torch.tensor([0.5, 0.3, 0.8], dtype=torch.float)  # 回归任务
            else:
                labels = torch.tensor([0, 1, min(2, num_labels-1)], dtype=torch.long)  # 分类任务
            
            print(f"输入形状: {inputs['input_ids'].shape}")
            print(f"标签形状: {labels.shape}")
            print(f"标签值: {labels}")
            print(f"标签数据类型: {labels.dtype}")
            
            # 前向传播
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                print(f"输出logits形状: {outputs.logits.shape}")
                print(f"损失值: {outputs.loss}")
                
                # 检查logits和labels的维度匹配
                logits_flat = outputs.logits.view(-1, model.num_labels)
                labels_flat = labels.view(-1)
                
                print(f"展平后logits形状: {logits_flat.shape}")
                print(f"展平后labels形状: {labels_flat.shape}")
                
                # 检查标签值是否在有效范围内
                if labels.dtype == torch.long:
                    if labels_flat.max() >= model.num_labels or labels_flat.min() < 0:
                        print(f"❌ 标签值超出范围! 标签范围: {labels_flat.min()}-{labels_flat.max()}, 期望范围: 0-{model.num_labels-1}")
                    else:
                        print(f"✅ 标签值在有效范围内")
                
                print("✅ 模型前向传播成功")
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_cross_entropy_loss():
    """测试交叉熵损失函数"""
    print("\n测试交叉熵损失函数...")
    
    try:
        import torch.nn as nn
        
        # 测试不同的配置
        test_cases = [
            (3, torch.tensor([0, 1, 2])),  # 正常情况
            (3, torch.tensor([0, 1, 3])),  # 标签超出范围
            (3, torch.tensor([-1, 0, 1])), # 负标签
            (2, torch.tensor([0, 1, 2])),  # 标签超出范围
        ]
        
        for num_classes, labels in test_cases:
            print(f"\n测试: num_classes={num_classes}, labels={labels}")
            
            # 创建随机logits
            logits = torch.randn(3, num_classes)
            
            # 创建损失函数
            criterion = nn.CrossEntropyLoss()
            
            try:
                loss = criterion(logits, labels)
                print(f"✅ 损失计算成功: {loss}")
            except Exception as e:
                print(f"❌ 损失计算失败: {e}")
                
    except Exception as e:
        print(f"❌ 交叉熵测试失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== 模型配置和标签匹配测试 ===")
    
    # 测试模型配置
    model_ok = test_model_config()
    
    # 测试交叉熵损失
    loss_ok = test_cross_entropy_loss()
    
    if model_ok and loss_ok:
        print("\n✅ 所有测试通过")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)

