import os
import sys
import torch
import logging
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_training_step():
    """模拟训练步骤"""
    print("模拟训练步骤...")
    
    try:
        # 导入必要的模块
        sys.path.append('/home/szk_25/FedSA-LoRA-Dual')
        sys.path.append('/home/szk_25/FedSA-LoRA')
        
        from dual_lora_model_builder import get_dual_lora_llm
        from federatedscope.core.configs.config import CN
        from federatedscope.glue.dataloader.dataloader import load_glue_dataset
        
        # 创建配置
        config = CN()
        config.model = CN()
        config.model.type = "/home/szk_25/FederatedLLM/llama-7b@huggingface_llm"
        config.data = CN()
        config.data.type = "mnli@glue"
        config.data.root = "/home/szk_25/FedSA-LoRA-Dual/GLUE"
        config.data.matched = True
        config.llm = CN()
        config.llm.tok_len = 128
        config.llm.cache = CN()
        config.llm.cache.model = "/home/szk_25/FederatedLLM/llama-7b"
        config.llm.adapter = CN()
        config.llm.adapter.use = True
        config.llm.adapter.args = [{
            'adapter_package': 'dual_lora',
            'adapter_method': 'dual_lora',
            'use_dual_lora': True,
            'global_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'local_r': 4,
            'fusion_method': 'weighted_sum',
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        }]
        
        # 添加federate配置
        config.federate = CN()
        config.federate.freeze_global = False
        config.federate.method = "fedavg"
        
        # 加载数据集（这会设置num_labels）
        print("加载数据集...")
        datasets = load_glue_dataset(config)
        if len(datasets) == 2:
            train_dataset, eval_dataset = datasets
            test_dataset = eval_dataset  # 使用验证集作为测试集
        else:
            train_dataset, eval_dataset, test_dataset = datasets
        
        print(f"配置的num_labels: {config.data.num_labels}")
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(eval_dataset)}")
        print(f"数据集num_labels: {config.data.num_labels}")
        
        # 检查数据集结构
        print(f"训练集类型: {type(train_dataset)}")
        if isinstance(train_dataset, tuple):
            print(f"训练集是tuple，长度: {len(train_dataset)}")
            if len(train_dataset) > 0:
                actual_dataset = train_dataset[0]
                print(f"实际数据集类型: {type(actual_dataset)}")
                if hasattr(actual_dataset, '__getitem__'):
                    sample = actual_dataset[0]
                    print(f"样本键: {sample.keys()}")
                    print(f"样本标签: {sample.get('label', 'No label')}")
        elif hasattr(train_dataset, '__getitem__'):
            sample = train_dataset[0]
            print(f"样本键: {sample.keys()}")
            print(f"样本标签: {sample.get('label', 'No label')}")
        
        # 获取实际的Dataset对象
        if isinstance(train_dataset, tuple):
            actual_train_dataset = train_dataset[0]
            actual_eval_dataset = eval_dataset[0] if isinstance(eval_dataset, tuple) else eval_dataset
        else:
            actual_train_dataset = train_dataset
            actual_eval_dataset = eval_dataset
            
        print(f"实际训练集大小: {len(actual_train_dataset)}")
        print(f"实际验证集大小: {len(actual_eval_dataset)}")
        
        # 创建数据加载器
        train_loader = DataLoader(actual_train_dataset, batch_size=2, shuffle=False)
        
        # 创建模型
        print("创建双模块LoRA模型...")
        model = get_dual_lora_llm(config)
        
        print(f"模型类型: {type(model)}")
        print(f"模型num_labels: {getattr(model, 'num_labels', 'Not found')}")
        
        # 检查模型的分类器
        if hasattr(model, 'score'):
            print(f"模型score层输出维度: {model.score.out_features}")
        elif hasattr(model, 'classifier'):
            print(f"模型classifier层输出维度: {model.classifier.out_features}")
        else:
            print("未找到分类器层")
        
        # 模拟训练步骤
        print("\n模拟训练步骤...")
        model.train()
        
        for i, batch in enumerate(train_loader):
            print(f"\n批次 {i}:")
            print(f"批次键: {batch.keys()}")
            
            # 检查标签
            if 'label' in batch:
                labels = batch['label']
                print(f"标签形状: {labels.shape}")
                print(f"标签值: {labels}")
                print(f"标签数据类型: {labels.dtype}")
                print(f"标签范围: {labels.min()} - {labels.max()}")
                
                # 检查标签是否超出范围
                if hasattr(model, 'num_labels'):
                    model_num_labels = model.num_labels
                elif hasattr(model, 'score'):
                    model_num_labels = model.score.out_features
                elif hasattr(model, 'classifier'):
                    model_num_labels = model.classifier.out_features
                else:
                    model_num_labels = config.data.num_labels
                
                print(f"模型期望的num_labels: {model_num_labels}")
                
                if labels.max() >= model_num_labels or labels.min() < 0:
                    print(f"❌ 标签值超出范围! 标签范围: {labels.min()}-{labels.max()}, 模型期望: 0-{model_num_labels-1}")
                    return False
                else:
                    print(f"✅ 标签值在有效范围内")
            
            # 前向传播
            try:
                outputs = model(**batch)
                print(f"输出logits形状: {outputs.logits.shape}")
                print(f"损失值: {outputs.loss}")
                
                # 检查logits和labels的维度匹配
                logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
                labels_flat = labels.view(-1)
                
                print(f"展平后logits形状: {logits_flat.shape}")
                print(f"展平后labels形状: {labels_flat.shape}")
                
                print("✅ 前向传播成功")
                
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 只测试第一个批次
            break
        
        print("✅ 训练步骤模拟成功")
        return True
        
    except Exception as e:
        print(f"❌ 模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 训练步骤模拟 ===")
    
    success = simulate_training_step()
    
    if success:
        print("\n✅ 模拟成功")
    else:
        print("\n❌ 模拟失败")
        sys.exit(1)
