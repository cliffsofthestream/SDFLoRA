#!/usr/bin/env python3


import sys
import os
import yaml
import torch
sys.path.append('/home/szk_25/FedSA-LoRA')
sys.path.append('/home/szk_25/FedSA-LoRA-Dual')

from federatedscope.core.configs.config import CN
from federatedscope.core.cmd_args import parse_args
from dual_lora_model_builder import get_dual_lora_llm

def verify_config_and_model(config_path='/home/szk_25/FedSA-LoRA-Dual/dual_lora_config.yaml'):
    """验证配置和模型是否正确设置"""
    print("=" * 60)
    print("验证配置和模型设置")
    print("=" * 60)
    
    # 1. 读取YAML配置
    print("\n1. 读取配置文件...")
    with open(config_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    print(f"   Method: {yaml_cfg['federate']['method']}")
    print(f"   Data num_labels (YAML): {yaml_cfg['data'].get('num_labels', 'Not set')}")
    
    # 2. 加载FederatedScope配置
    print("\n2. 加载FederatedScope配置...")
    # 使用parse_args加载配置（模拟命令行参数）
    import sys
    original_argv = sys.argv
    sys.argv = ['verify_config_and_model.py', '--cfg', config_path]
    try:
        from federatedscope.core.configs.config import global_cfg
        args = parse_args()
        cfg = global_cfg.clone()
        cfg.merge_from_file(config_path)
    finally:
        sys.argv = original_argv
    
    print(f"   Method: {cfg.federate.method}")
    print(f"   Data num_labels: {getattr(cfg.data, 'num_labels', 'Not set')}")
    
    # 确保num_labels被设置
    if not hasattr(cfg.data, 'num_labels') or cfg.data.num_labels is None:
        cfg.data.num_labels = 3
        print(f"   ⚠️  num_labels未设置，设置为默认值: 3")
    
    # 3. 加载数据集以验证标签
    print("\n3. 验证数据集标签...")
    try:
        from federatedscope.glue.dataloader.dataloader import load_glue_dataset
        dataset, updated_config = load_glue_dataset(cfg)
        train_dataset, eval_dataset, test_dataset = dataset
        
        if hasattr(train_dataset, 'label'):
            train_labels = train_dataset['label']
            min_label = train_labels.min().item() if hasattr(train_labels, 'min') else min(train_labels)
            max_label = train_labels.max().item() if hasattr(train_labels, 'max') else max(train_labels)
            print(f"   训练集标签范围: {min_label} - {max_label}")
            
            if max_label >= cfg.data.num_labels:
                print(f"   ❌ 错误: 标签值 {max_label} >= num_labels {cfg.data.num_labels}")
                print(f"   💡 建议: 确保num_labels设置为 {max_label + 1} 或更大")
            else:
                print(f"   ✅ 标签范围在有效范围内 [0, {cfg.data.num_labels})")
        else:
            print("   ⚠️  无法访问数据集标签")
            
    except Exception as e:
        print(f"   ⚠️  无法加载数据集: {e}")
    
    # 4. 创建模型并验证分类器
    print("\n4. 验证模型分类器...")
    try:
        model = get_dual_lora_llm(cfg)
        
        # 查找分类器层
        classifier = None
        classifier_name = None
        
        # 检查不同的可能分类器名称
        for name, module in model.named_modules():
            if hasattr(module, 'out_features') or name in ['classifier', 'score', 'head']:
                if hasattr(module, 'weight'):
                    classifier = module
                    classifier_name = name
                    break
        
        if classifier is None:
            # 尝试从model的属性获取
            for attr_name in ['classifier', 'score', 'head']:
                if hasattr(model, attr_name):
                    classifier = getattr(model, attr_name)
                    classifier_name = attr_name
                    break
        
        if classifier is None:
            print("   ⚠️  未找到分类器层")
        else:
            if hasattr(classifier, 'out_features'):
                model_num_labels = classifier.out_features
                print(f"   分类器层 ({classifier_name}): out_features = {model_num_labels}")
                
                if model_num_labels != cfg.data.num_labels:
                    print(f"   ❌ 错误: 模型分类器输出维度 ({model_num_labels}) != 配置的num_labels ({cfg.data.num_labels})")
                    print(f"   💡 建议: 确保配置中的num_labels设置为 {model_num_labels}")
                else:
                    print(f"   ✅ 模型分类器输出维度匹配配置: {model_num_labels}")
            else:
                print(f"   ⚠️  分类器层 {classifier_name} 没有 out_features 属性")
                print(f"   分类器权重形状: {classifier.weight.shape if hasattr(classifier, 'weight') else 'N/A'}")
        
        # 5. 测试前向传播
        print("\n5. 测试模型前向传播...")
        try:
            # 创建虚拟输入
            batch_size = 2
            seq_len = 128
            input_ids = torch.randint(1, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            labels = torch.randint(0, cfg.data.num_labels, (batch_size,))
            
            print(f"   输入形状: {input_ids.shape}")
            print(f"   标签值: {labels.tolist()}")
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, 
                              attention_mask=attention_mask,
                              labels=labels)
            
            if hasattr(outputs, 'logits'):
                print(f"   ✅ 前向传播成功")
                print(f"   Logits形状: {outputs.logits.shape}")
                print(f"   预期形状: ({batch_size}, {cfg.data.num_labels})")
                
                if outputs.logits.shape[1] != cfg.data.num_labels:
                    print(f"   ❌ 错误: Logits输出维度不匹配")
                else:
                    print(f"   ✅ Logits输出维度匹配")
            else:
                print(f"   ⚠️  输出没有logits属性")
                
        except Exception as e:
            print(f"   ❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"   ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)

if __name__ == "__main__":
    verify_config_and_model()

