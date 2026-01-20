#!/usr/bin/env python3
"""
调试实际训练过程中的标签问题
"""

import os
import sys
import torch
import logging
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_training_step():
    """调试训练步骤"""
    print("调试训练步骤...")
    
    try:
        # 导入必要的模块
        sys.path.append('/home/user/FedSA-LoRA-Dual')
        sys.path.append('/home/user/FedSA-LoRA')
        
        from code.dual_lora_model_builder import get_dual_lora_llm
        from federatedscope.core.configs.config import CN
        
        # 创建配置
        config = CN()
        config.model = CN()
        config.model.type = "/home/user/FederatedLLM/llama-7b@huggingface_llm"
        config.data = CN()
        config.data.num_labels = 3
        config.data.type = "mnli@glue"
        config.data.root = "/home/user/FedSA-LoRA-Dual/GLUE"
        config.data.matched = True
        config.llm = CN()
        config.llm.tok_len = 128
        config.llm.cache = CN()
        config.llm.cache.model = "/home/user/FederatedLLM/llama-7b"
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
        
        print(f"配置的num_labels: {config.data.num_labels}")
        
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
        
        # 创建测试数据
        tokenizer = AutoTokenizer.from_pretrained("/home/user/FederatedLLM/llama-7b")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_texts = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "Birds fly in the sky."
        ]
        
        inputs = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 创建标签
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        
        print(f"输入形状: {inputs['input_ids'].shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels}")
        
        # 前向传播
        print("执行前向传播...")
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            print(f"输出logits形状: {outputs.logits.shape}")
            print(f"损失值: {outputs.loss}")
            
            # 检查logits和labels的维度匹配
            logits_flat = outputs.logits.view(-1, outputs.logits.size(-1))
            labels_flat = labels.view(-1)
            
            print(f"展平后logits形状: {logits_flat.shape}")
            print(f"展平后labels形状: {labels_flat.shape}")
            print(f"模型期望的num_labels: {getattr(model, 'num_labels', 'Unknown')}")
            print(f"实际logits的最后一维: {outputs.logits.size(-1)}")
            
            # 检查标签值是否在有效范围内
            if labels_flat.max() >= outputs.logits.size(-1) or labels_flat.min() < 0:
                print(f"❌ 标签值超出范围! 标签范围: {labels_flat.min()}-{labels_flat.max()}, logits维度: {outputs.logits.size(-1)}")
                return False
            else:
                print(f"✅ 标签值在有效范围内")
        
        print("✅ 模型前向传播成功")
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 训练步骤调试 ===")
    
    success = debug_training_step()
    
    if success:
        print("\n✅ 调试成功")
    else:
        print("\n❌ 调试失败")
        sys.exit(1)
