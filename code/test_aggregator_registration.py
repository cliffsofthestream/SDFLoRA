
import sys
import os

# 添加路径
ORIGINAL_PROJECT_PATH = "/home/user/FedSA-LoRA"
if ORIGINAL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_PATH)

CURRENT_PROJECT_PATH = "/home/user/FedSA-LoRA-Dual"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)

# 导入模块
from code.dual_lora_aggregator import DualLoRAFederatedAggregator

def test_aggregator_registration():
    """测试聚合器注册"""
    try:
        from federatedscope.register import register_aggregator
        register_aggregator('dual_lora_aggregator', DualLoRAFederatedAggregator)
        print("✓ 聚合器注册成功")
        
        # 检查注册是否成功
        from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
        aggregator_class = get_aggregator('dual_lora_aggregator')
        if aggregator_class == DualLoRAFederatedAggregator:
            print("✓ 聚合器注册验证成功")
        else:
            print("✗ 聚合器注册验证失败")
            
    except Exception as e:
        print(f"✗ 聚合器注册失败: {e}")

if __name__ == "__main__":
    test_aggregator_registration()
