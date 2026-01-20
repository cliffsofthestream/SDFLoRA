import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from dp_sgd_engine import DPSGDConfig, DualLoRADPSGDTrainer

def test_noise_addition():
    """测试噪声是否被添加到梯度中"""
    logger.info("测试噪声添加...")
    
    # 创建简单测试模型
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_lora_A = nn.Linear(10, 5, bias=False)
            self.global_lora_B = nn.Linear(5, 10, bias=False)
            self.local_lora_A = nn.Linear(10, 3, bias=False)
            self.local_lora_B = nn.Linear(3, 10, bias=False)
        
        def forward(self, x):
            return x
    
    model = TestModel()
    
    # 创建DP-SGD配置
    config = DPSGDConfig(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        apply_to_global=True,
        apply_to_local=False
    )
    
    trainer = DualLoRADPSGDTrainer(model, config)
    
    # 记录原始参数
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.clone()
    
    # 添加随机梯度
    for name, param in model.named_parameters():
        if 'global_lora' in name:
            param.grad = torch.randn_like(param) * 0.1
    
    # 记录添加噪声前的梯度
    grads_before = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_before[name] = param.grad.clone()
    
    # 添加噪声
    trainer.add_noise_to_gradients(model, noise_scale=1.0)
    
    # 记录添加噪声后的梯度
    grads_after = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_after[name] = param.grad.clone()
    
    # 检查噪声是否被添加
    noise_added = False
    noise_stats = {}
    
    for name in grads_before.keys():
        if name in grads_after:
            grad_before = grads_before[name]
            grad_after = grads_after[name]
            
            if not torch.equal(grad_before, grad_after):
                noise_added = True
                noise = grad_after - grad_before
                noise_stats[name] = {
                    'mean': noise.mean().item(),
                    'std': noise.std().item(),
                    'max_abs': noise.abs().max().item()
                }
                logger.info(f"{name}: 噪声已添加, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
            else:
                logger.info(f"{name}: 未检测到噪声")
    
    return noise_added, noise_stats

def test_aggregation_noise():
    """测试聚合噪声"""
    logger.info("测试聚合噪声...")
    
    from dual_lora_aggregator import DualLoRAAggregator
    
    # 创建聚合器配置
    dp_config = {
        'enabled': True,
        'epsilon': 1.0,
        'delta': 1e-5,
        'max_grad_norm': 1.0,
        'enable_secure_aggregation': True,
        'aggregation_noise_scale': 0.8
    }
    
    aggregator = DualLoRAAggregator(enable_dp_sgd=True, dp_config=dp_config)
    
    # 创建测试参数
    test_params = {
        'global_lora_A.weight': torch.randn(5, 10),
        'global_lora_B.weight': torch.randn(10, 5),
        'local_lora_A.weight': torch.randn(3, 10),
        'local_lora_B.weight': torch.randn(10, 3)
    }
    
    # 添加聚合噪声
    noisy_params = aggregator._add_aggregation_noise(test_params, num_clients=3)
    
    # 检查噪声是否被添加
    noise_added = False
    for key in ['global_lora_A.weight', 'global_lora_B.weight']:
        if key in test_params and key in noisy_params:
            if not torch.equal(test_params[key], noisy_params[key]):
                noise_added = True
                noise = noisy_params[key] - test_params[key]
                logger.info(f"{key}: 聚合噪声已添加, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}")
            else:
                logger.info(f"{key}: 未检测到聚合噪声")
    
    return noise_added

def test_different_seeds():
    """测试不同随机种子的影响"""
    logger.info("测试不同随机种子的影响...")
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_lora_A = nn.Linear(10, 5, bias=False)
            self.global_lora_B = nn.Linear(5, 10, bias=False)
        
        def forward(self, x):
            return x
    
    seeds = [123, 456, 789, 999, 42]
    results = {}
    
    for seed in seeds:
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = TestModel()
        config = DPSGDConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        trainer = DualLoRADPSGDTrainer(model, config)
        
        # 添加随机梯度
        for name, param in model.named_parameters():
            if 'global_lora' in name:
                param.grad = torch.randn_like(param) * 0.1
        
        # 记录添加噪声前的梯度
        grads_before = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads_before[name] = param.grad.clone()
        
        # 添加噪声
        trainer.add_noise_to_gradients(model, noise_scale=1.0)
        
        # 记录添加噪声后的梯度
        grads_after = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads_after[name] = param.grad.clone()
        
        # 计算噪声
        noise_values = []
        for name in grads_before.keys():
            if name in grads_after:
                noise = grads_after[name] - grads_before[name]
                noise_values.extend(noise.flatten().tolist())
        
        results[seed] = {
            'mean': np.mean(noise_values) if noise_values else 0.0,
            'std': np.std(noise_values) if noise_values else 0.0,
            'count': len(noise_values)
        }
        
        logger.info(f"种子 {seed}: mean={results[seed]['mean']:.6f}, std={results[seed]['std']:.6f}, count={results[seed]['count']}")
    
    # 计算不同种子间的差异
    seed_means = [results[seed]['mean'] for seed in seeds]
    seed_stds = [results[seed]['std'] for seed in seeds]
    
    mean_variance = np.var(seed_means)
    std_variance = np.var(seed_stds)
    
    logger.info(f"不同种子间均值方差: {mean_variance:.6f}")
    logger.info(f"不同种子间标准差方差: {std_variance:.6f}")
    
    return results

def main():
    """主函数"""
    logger.info("开始差分隐私噪声测试...")
    
    # 1. 测试梯度噪声添加
    logger.info("=" * 50)
    noise_added, noise_stats = test_noise_addition()
    logger.info(f"梯度噪声添加: {'成功' if noise_added else '失败'}")
    
    # 2. 测试聚合噪声
    logger.info("=" * 50)
    agg_noise_added = test_aggregation_noise()
    logger.info(f"聚合噪声添加: {'成功' if agg_noise_added else '失败'}")
    
    # 3. 测试不同种子的影响
    logger.info("=" * 50)
    seed_results = test_different_seeds()
    
    # 总结
    logger.info("=" * 50)
    logger.info("测试总结:")
    logger.info(f"1. 梯度噪声添加: {'✓' if noise_added else '✗'}")
    logger.info(f"2. 聚合噪声添加: {'✓' if agg_noise_added else '✗'}")
    logger.info(f"3. 不同种子产生不同噪声: {'✓' if len(set(str(seed_results[s]['mean']) for s in seed_results)) > 1 else '✗'}")
    
    if noise_added and agg_noise_added:
        logger.info("✓ 差分隐私噪声机制正常工作!")
    else:
        logger.warning("✗ 差分隐私噪声机制存在问题!")

if __name__ == "__main__":
    main()
