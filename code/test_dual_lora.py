
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 添加项目路径
CURRENT_PROJECT_PATH = "/home/szk_25/FedSA-LoRA-Dual"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)

from code.dual_lora_adapter import DualLoRALayer, DualLoRAModel, DualLoRAConfig, create_dual_lora_model
from code.dual_lora_peft_adapter import DualLoraConfig, DualLoraPeftModel, enable_dual_lora_adapter
from code.dual_lora_aggregator import DualLoRAAggregator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dual_lora_layer():
    """测试双模块LoRA层"""
    logger.info("Testing DualLoRALayer...")
    
    # 创建测试数据
    batch_size, seq_len, hidden_dim = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建双模块LoRA层
    dual_lora = DualLoRALayer(
        in_features=hidden_dim,
        out_features=hidden_dim,
        global_rank=8,
        local_rank=4,
        fusion_method="weighted_sum"
    )
    
    # 前向传播
    output = dual_lora(x)
    
    # 验证输出形状
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    
    # 验证参数获取
    global_params = dual_lora.get_global_parameters()
    local_params = dual_lora.get_local_parameters()
    
    assert len(global_params) == 2, f"Expected 2 global parameters, got {len(global_params)}"
    assert len(local_params) >= 2, f"Expected at least 2 local parameters, got {len(local_params)}"
    
    logger.info("✓ DualLoRALayer test passed")
    return True


def test_dual_lora_aggregator():
    """测试双模块LoRA聚合器"""
    logger.info("Testing DualLoRAAggregator...")
    
    # 创建模拟客户端参数
    client_params = []
    for i in range(3):
        params = {
            f"layer.global_lora_A.weight": torch.randn(8, 768),
            f"layer.global_lora_B.weight": torch.randn(768, 8),
            f"layer.local_lora_A.weight": torch.randn(4, 768),
            f"layer.local_lora_B.weight": torch.randn(768, 4),
            f"classifier.weight": torch.randn(2, 768),
        }
        client_params.append(params)
    
    # 创建聚合器
    aggregator = DualLoRAAggregator(
        global_aggregation_strategy="fedavg",
        local_personalization_strategy="local_only",
        client_ranks={1: (8, 4), 2: (8, 4), 3: (8, 4)}
    )
    
    # 模拟聚合信息
    agg_info = {
        "client_feedback": [
            (1, (100, client_params[0])),
            (2, (150, client_params[1])),
            (3, (120, client_params[2]))
        ]
    }
    
    # 执行聚合
    aggregated = aggregator.aggregate(agg_info)
    
    # 验证聚合结果
    assert isinstance(aggregated, dict), "Aggregated result should be a dictionary"
    
    # 验证全局参数被聚合
    global_param_count = sum(1 for k in aggregated.keys() if "global_lora" in k)
    assert global_param_count > 0, "No global parameters found in aggregated result"
    
    logger.info("✓ DualLoRAAggregator test passed")
    return True


def test_fusion_methods():
    """测试不同的融合方法"""
    logger.info("Testing fusion methods...")
    
    batch_size, seq_len, hidden_dim = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    fusion_methods = ["weighted_sum", "gating", "attention"]
    
    for method in fusion_methods:
        logger.info(f"Testing fusion method: {method}")
        
        try:
            dual_lora = DualLoRALayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                global_rank=8,
                local_rank=4,
                fusion_method=method
            )
            
            output = dual_lora(x)
            assert output.shape == x.shape, f"Output shape mismatch for {method}"
            
            logger.info(f"✓ Fusion method {method} test passed")
            
        except Exception as e:
            logger.error(f"✗ Fusion method {method} test failed: {e}")
            return False
    
    return True


def test_heterogeneous_ranks():
    """测试异构rank配置"""
    logger.info("Testing heterogeneous ranks...")
    
    # 创建不同rank的客户端参数
    client_configs = [
        {"global_r": 12, "local_r": 4},
        {"global_r": 8, "local_r": 8},
        {"global_r": 4, "local_r": 12}
    ]
    
    client_params = []
    for i, config in enumerate(client_configs):
        params = {
            f"layer.global_lora_A.weight": torch.randn(config["global_r"], 768),
            f"layer.global_lora_B.weight": torch.randn(768, config["global_r"]),
            f"layer.local_lora_A.weight": torch.randn(config["local_r"], 768),
            f"layer.local_lora_B.weight": torch.randn(768, config["local_r"]),
        }
        client_params.append(params)
    
    # 创建异构聚合器
    aggregator = DualLoRAAggregator(
        global_aggregation_strategy="stacked",
        enable_stacking=True,
        enable_heterogeneous=True,
        client_ranks={1: (12, 4), 2: (8, 8), 3: (4, 12)}
    )
    
    # 模拟聚合
    agg_info = {
        "client_feedback": [
            (1, (100, client_params[0])),
            (2, (150, client_params[1])),
            (3, (120, client_params[2]))
        ]
    }
    
    try:
        aggregated = aggregator.aggregate(agg_info)
        logger.info("✓ Heterogeneous ranks test passed")
        return True
    except Exception as e:
        logger.error(f"✗ Heterogeneous ranks test failed: {e}")
        return False


def test_parameter_separation():
    """测试参数分离功能"""
    logger.info("Testing parameter separation...")
    
    # 创建混合参数字典
    mixed_params = {
        "layer1.global_lora_A.weight": torch.randn(8, 768),
        "layer1.global_lora_B.weight": torch.randn(768, 8),
        "layer1.local_lora_A.weight": torch.randn(4, 768),
        "layer1.local_lora_B.weight": torch.randn(768, 4),
        "layer1.global_weight": torch.tensor(0.7),
        "layer1.local_weight": torch.tensor(0.3),
        "classifier.weight": torch.randn(2, 768),
        "base_model.embedding.weight": torch.randn(30000, 768),
    }
    
    aggregator = DualLoRAAggregator()
    
    # 测试参数分离
    global_params, local_params, other_params = aggregator._separate_parameters([mixed_params])
    
    global_count = len(global_params[0])
    local_count = len(local_params[0])
    other_count = len(other_params[0])
    
    logger.info(f"Global params: {global_count}, Local params: {local_count}, Other params: {other_count}")
    
    # 验证分离结果
    assert global_count == 2, f"Expected 2 global params, got {global_count}"
    assert local_count == 4, f"Expected 4 local params, got {local_count}"
    assert other_count == 2, f"Expected 2 other params, got {other_count}"
    
    logger.info("✓ Parameter separation test passed")
    return True


def test_model_state_dict():
    """测试模型状态字典功能"""
    logger.info("Testing model state dict...")
    
    # 创建简单的基础模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 768)
            self.classifier = nn.Linear(768, 2)
        
        def forward(self, x):
            x = self.linear(x)
            return self.classifier(x)
    
    base_model = SimpleModel()
    
    # 创建双模块LoRA配置
    config = DualLoRAConfig(
        global_rank=8,
        local_rank=4,
        target_modules=["linear"]
    )
    
    # 创建双模块LoRA模型
    dual_model = DualLoRAModel(base_model, config)
    
    # 测试状态字典
    global_state = dual_model.get_global_state_dict()
    local_state = dual_model.get_local_state_dict()
    full_state = dual_model.state_dict()
    
    logger.info(f"Global state keys: {len(global_state)}")
    logger.info(f"Local state keys: {len(local_state)}")
    logger.info(f"Full state keys: {len(full_state)}")
    
    # 验证状态字典
    assert len(global_state) > 0, "Global state dict should not be empty"
    assert len(local_state) > 0, "Local state dict should not be empty"
    assert len(full_state) >= len(global_state) + len(local_state), "Full state dict size mismatch"
    
    logger.info("✓ Model state dict test passed")
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("Starting Dual-LoRA tests...")
    
    tests = [
        test_dual_lora_layer,
        test_dual_lora_aggregator,
        test_fusion_methods,
        test_heterogeneous_ranks,
        test_parameter_separation,
        test_model_state_dict,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            failed += 1
    
    logger.info(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
