

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 添加项目路径
CURRENT_PROJECT_PATH = "/home/user/FedSA-LoRA-Dual"
ORIGINAL_PROJECT_PATH = "/home/user/FedSA-LoRA"

if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)
if ORIGINAL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_PATH)

from code.dual_lora_adapter import create_dual_lora_model, DualLoRAConfig
from code.dual_lora_peft_adapter import enable_dual_lora_adapter, DualLoraConfig
from code.dual_lora_aggregator import DualLoRAAggregator
from code.dual_lora_model_builder import DualLoRAModelBuilder

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_usage():
    """示例1: 基础使用方法"""
    logger.info("=== 示例1: 基础使用方法 ===")
    
    # 创建一个简单的基础模型
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_dim=768, num_classes=2):
            super().__init__()
            self.embedding = nn.Embedding(30000, hidden_dim)
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.dense = nn.Linear(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            # 简化的注意力机制
            q = self.query(x)
            k = self.key(x)
            v = self.value(x)
            # 简化处理
            attn_output = self.dense(v)
            # 池化并分类
            pooled = attn_output.mean(dim=1)
            return self.classifier(pooled)
    
    # 创建基础模型
    base_model = SimpleTransformer()
    logger.info(f"Base model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # 应用双模块LoRA
    dual_lora_model = enable_dual_lora_adapter(
        model=base_model,
        global_r=8,
        local_r=4,
        lora_alpha=16,
        fusion_method="weighted_sum",
        target_modules=["query", "key", "value", "dense"]
    )
    
    # 打印参数信息
    dual_lora_model.print_trainable_parameters()
    
    # 测试前向传播
    input_ids = torch.randint(0, 30000, (2, 10))
    output = dual_lora_model(input_ids)
    logger.info(f"Output shape: {output.shape}")
    
    # 获取全局和本地参数
    global_params = dual_lora_model.get_global_state_dict()
    local_params = dual_lora_model.get_local_state_dict()
    
    logger.info(f"Global parameters: {len(global_params)} tensors")
    logger.info(f"Local parameters: {len(local_params)} tensors")
    
    return dual_lora_model


def example_2_federated_aggregation():
    """示例2: 联邦聚合过程"""
    logger.info("\n=== 示例2: 联邦聚合过程 ===")
    
    # 模拟3个客户端的双模块LoRA参数
    client_models = []
    
    for client_id in range(1, 4):
        # 每个客户端有不同的rank配置
        global_r = 8 if client_id == 1 else (6 if client_id == 2 else 4)
        local_r = 4 if client_id == 1 else (6 if client_id == 2 else 8)
        
        logger.info(f"Client {client_id}: global_r={global_r}, local_r={local_r}")
        
        # 模拟客户端参数
        client_params = {
            f"query.global_lora_A.weight": torch.randn(global_r, 768),
            f"query.global_lora_B.weight": torch.randn(768, global_r),
            f"query.local_lora_A.weight": torch.randn(local_r, 768),
            f"query.local_lora_B.weight": torch.randn(768, local_r),
            f"query.global_weight": torch.tensor(0.7),
            f"query.local_weight": torch.tensor(0.3),
            f"classifier.weight": torch.randn(2, 768),
            f"classifier.bias": torch.randn(2),
        }
        client_models.append(client_params)
    
    # 创建聚合器
    aggregator = DualLoRAAggregator(
        global_aggregation_strategy="stacked",  # 使用堆叠聚合支持异构
        local_personalization_strategy="local_only",
        client_ranks={1: (8, 4), 2: (6, 6), 3: (4, 8)},
        enable_stacking=True,
        enable_heterogeneous=True
    )
    
    # 准备聚合信息
    agg_info = {
        "client_feedback": [
            (1, (100, client_models[0])),  # (client_id, (sample_size, model_params))
            (2, (150, client_models[1])),
            (3, (120, client_models[2]))
        ]
    }
    
    # 执行聚合
    aggregated_params = aggregator.aggregate(agg_info)
    
    logger.info(f"Aggregated parameters: {len(aggregated_params)} tensors")
    
    # 显示聚合后的全局参数形状
    for key, value in aggregated_params.items():
        if "global_lora" in key:
            logger.info(f"{key}: {value.shape}")
    
    return aggregated_params


def example_3_heterogeneous_clients():
    """示例3: 异构客户端配置"""
    logger.info("\n=== 示例3: 异构客户端配置 ===")
    
    # 定义不同类型的客户端配置
    client_configs = {
        "high_resource": {"global_r": 16, "local_r": 4},      # 高资源客户端
        "balanced": {"global_r": 8, "local_r": 8},            # 平衡客户端
        "personalized": {"global_r": 4, "local_r": 16},       # 高个性化客户端
        "limited": {"global_r": 4, "local_r": 4},             # 资源受限客户端
    }
    
    models = {}
    
    for client_type, config in client_configs.items():
        logger.info(f"Creating {client_type} client model...")
        
        # 创建基础模型
        base_model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )
        
        # 应用双模块LoRA
        dual_model = enable_dual_lora_adapter(
            model=base_model,
            global_r=config["global_r"],
            local_r=config["local_r"],
            fusion_method="gating",  # 使用门控融合
            target_modules=["0", "2"]  # 目标线性层
        )
        
        models[client_type] = dual_model
        dual_model.print_trainable_parameters()
    
    return models


def example_4_fusion_methods_comparison():
    """示例4: 融合方法比较"""
    logger.info("\n=== 示例4: 融合方法比较 ===")
    
    fusion_methods = ["weighted_sum", "gating", "attention"]
    
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 4, 20, 768
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    results = {}
    
    for method in fusion_methods:
        logger.info(f"Testing fusion method: {method}")
        
        # 创建基础模型
        base_model = nn.Linear(hidden_dim, hidden_dim)
        
        # 应用双模块LoRA
        dual_model = enable_dual_lora_adapter(
            model=base_model,
            global_r=8,
            local_r=4,
            fusion_method=method,
            target_modules=[""]  # 应用到整个模型
        )
        
        # 前向传播
        with torch.no_grad():
            output = dual_model(x)
        
        results[method] = {
            "output_shape": output.shape,
            "output_mean": output.mean().item(),
            "output_std": output.std().item()
        }
        
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Output mean: {output.mean().item():.4f}")
        logger.info(f"  Output std: {output.std().item():.4f}")
    
    return results


def example_5_parameter_analysis():
    """示例5: 参数分析"""
    logger.info("\n=== 示例5: 参数分析 ===")
    
    # 创建不同配置的模型进行比较
    configs = [
        {"name": "Small", "global_r": 4, "local_r": 2},
        {"name": "Medium", "global_r": 8, "local_r": 4},
        {"name": "Large", "global_r": 16, "local_r": 8},
        {"name": "Global-focused", "global_r": 16, "local_r": 2},
        {"name": "Local-focused", "global_r": 4, "local_r": 16},
    ]
    
    base_model = nn.Sequential(
        nn.Linear(768, 768),
        nn.Linear(768, 768),
        nn.Linear(768, 2)
    )
    
    base_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Base model parameters: {base_params:,}")
    
    for config in configs:
        dual_model = enable_dual_lora_adapter(
            model=nn.Sequential(
                nn.Linear(768, 768),
                nn.Linear(768, 768),
                nn.Linear(768, 2)
            ),
            global_r=config["global_r"],
            local_r=config["local_r"],
            target_modules=["0", "1"]
        )
        
        global_params = sum(p.numel() for p in dual_model.get_global_state_dict().values())
        local_params = sum(p.numel() for p in dual_model.get_local_state_dict().values())
        total_trainable = global_params + local_params
        
        logger.info(f"{config['name']} configuration:")
        logger.info(f"  Global params: {global_params:,}")
        logger.info(f"  Local params: {local_params:,}")
        logger.info(f"  Total trainable: {total_trainable:,}")
        logger.info(f"  Efficiency: {total_trainable/base_params*100:.2f}%")


def main():
    """运行所有示例"""
    logger.info("双模块LoRA使用示例")
    logger.info("=" * 50)
    
    try:
        # 运行所有示例
        model1 = example_1_basic_usage()
        agg_params = example_2_federated_aggregation()
        hetero_models = example_3_heterogeneous_clients()
        fusion_results = example_4_fusion_methods_comparison()
        example_5_parameter_analysis()
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 所有示例运行成功！")
        logger.info("双模块LoRA实现验证完成。")
        
        return True
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
