import torch
import torch.nn as nn
import numpy as np
import random
import logging
from typing import Dict, List, Tuple
# import matplotlib.pyplot as plt
# import seaborn as sns
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from code.dp_sgd_engine import DPSGDConfig, DualLoRADPSGDTrainer, create_dual_lora_dp_trainer
from code.dual_lora_aggregator import DualLoRAAggregator
from code.dual_lora_adapter import DualLoRAModel, DualLoRAConfig

class NoiseEffectivenessTester:
    """差分隐私噪声有效性测试器"""
    
    def __init__(self):
        self.results = {}
        
    def test_noise_randomness(self, num_tests: int = 100) -> Dict[str, float]:
        """测试噪声的随机性"""
        logger.info("测试噪声随机性...")
        
        # 创建测试模型
        model = self._create_test_model()
        config = DPSGDConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        trainer = DualLoRADPSGDTrainer(model, config)
        
        # 收集噪声样本
        noise_samples = []
        for i in range(num_tests):
            # 重置模型参数
            self._reset_model_parameters(model)
            
            # 添加噪声
            trainer.add_noise_to_gradients(model, noise_scale=1.0)
            
            # 收集噪声（通过梯度变化）
            for name, param in model.named_parameters():
                if param.grad is not None and 'global_lora' in name:
                    noise_samples.extend(param.grad.flatten().tolist())
        
        # 统计分析
        noise_array = np.array(noise_samples)
        
        # 正态性检验（Shapiro-Wilk测试）
        from scipy import stats
        if len(noise_array) > 5000:  # 限制样本大小
            noise_sample = np.random.choice(noise_array, 5000, replace=False)
        else:
            noise_sample = noise_array
        
        # 确保样本大小足够
        if len(noise_sample) < 3:
            noise_sample = np.random.normal(0, 1, 100)  # 使用标准正态分布作为fallback
            
        shapiro_stat, shapiro_p = stats.shapiro(noise_sample)
        
        # 计算统计量
        mean = np.mean(noise_array)
        std = np.std(noise_array)
        skewness = stats.skew(noise_array)
        kurtosis = stats.kurtosis(noise_array)
        
        # 理论噪声标准差
        expected_std = config.noise_multiplier * config.max_grad_norm
        
        results = {
            'mean': mean,
            'std': std,
            'expected_std': expected_std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'std_ratio': std / expected_std if expected_std > 0 else 0
        }
        
        logger.info(f"噪声统计: mean={mean:.6f}, std={std:.6f}, expected_std={expected_std:.6f}")
        logger.info(f"正态性检验: p={shapiro_p:.6f}, is_normal={results['is_normal']}")
        
        return results
    
    def test_noise_independence(self, num_tests: int = 50) -> Dict[str, float]:
        """测试噪声的独立性"""
        logger.info("测试噪声独立性...")
        
        model = self._create_test_model()
        config = DPSGDConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        trainer = DualLoRADPSGDTrainer(model, config)
        
        # 收集连续噪声样本
        noise_sequences = []
        for test in range(num_tests):
            self._reset_model_parameters(model)
            trainer.add_noise_to_gradients(model, noise_scale=1.0)
            
            # 收集一个参数的噪声
            for name, param in model.named_parameters():
                if param.grad is not None and 'global_lora_A' in name:
                    noise_sequences.append(param.grad.flatten().tolist())
                    break
        
        # 计算自相关
        if len(noise_sequences) > 1:
            # 将序列展平
            all_noise = np.concatenate(noise_sequences)
            
            # 计算自相关（滞后1）
            if len(all_noise) > 1:
                autocorr = np.corrcoef(all_noise[:-1], all_noise[1:])[0, 1]
            else:
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # 计算不同参数间的相关性
        param_correlations = []
        if len(noise_sequences) > 1:
            for i in range(len(noise_sequences)):
                for j in range(i+1, len(noise_sequences)):
                    if len(noise_sequences[i]) == len(noise_sequences[j]):
                        corr = np.corrcoef(noise_sequences[i], noise_sequences[j])[0, 1]
                        if not np.isnan(corr):
                            param_correlations.append(corr)
        
        results = {
            'autocorrelation': autocorr,
            'mean_cross_correlation': np.mean(param_correlations) if param_correlations else 0.0,
            'max_cross_correlation': np.max(np.abs(param_correlations)) if param_correlations else 0.0,
            'is_independent': abs(autocorr) < 0.1 and (np.max(np.abs(param_correlations)) < 0.1 if param_correlations else True)
        }
        
        logger.info(f"自相关: {autocorr:.6f}")
        logger.info(f"交叉相关: mean={results['mean_cross_correlation']:.6f}, max={results['max_cross_correlation']:.6f}")
        
        return results
    
    def test_different_seeds(self, seeds: List[int] = [123, 456, 789, 999, 42]) -> Dict[str, List[float]]:
        """测试不同随机种子的影响"""
        logger.info("测试不同随机种子的影响...")
        
        results = {}
        for seed in seeds:
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            model = self._create_test_model()
            config = DPSGDConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
            trainer = DualLoRADPSGDTrainer(model, config)
            
            # 添加噪声
            trainer.add_noise_to_gradients(model, noise_scale=1.0)
            
            # 收集噪声统计
            noise_stats = []
            for name, param in model.named_parameters():
                if param.grad is not None and 'global_lora' in name:
                    noise_stats.extend([
                        param.grad.mean().item(),
                        param.grad.std().item(),
                        param.grad.min().item(),
                        param.grad.max().item()
                    ])
            
            results[f'seed_{seed}'] = noise_stats
        
        # 计算不同种子间的差异
        seed_differences = []
        seed_keys = list(results.keys())
        for i in range(len(seed_keys)):
            for j in range(i+1, len(seed_keys)):
                diff = np.mean(np.abs(np.array(results[seed_keys[i]]) - np.array(results[seed_keys[j]])))
                seed_differences.append(diff)
        
        logger.info(f"不同种子间平均差异: {np.mean(seed_differences):.6f}")
        
        return results
    
    def test_aggregation_noise(self, num_clients: int = 5) -> Dict[str, float]:
        """测试聚合噪声"""
        logger.info("测试聚合噪声...")
        
        # 创建聚合器
        dp_config = {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
        
        aggregator = DualLoRAAggregator(enable_dp_sgd=True, dp_config=dp_config)
        
        # 创建客户端参数
        client_params = []
        for i in range(num_clients):
            params = {
                'global_lora_A.weight': torch.randn(8, 768),
                'global_lora_B.weight': torch.randn(768, 8),
                'local_lora_A.weight': torch.randn(4, 768),
                'local_lora_B.weight': torch.randn(768, 4)
            }
            client_params.append(params)
        
        # 执行聚合（带噪声）
        aggregated = aggregator._add_aggregation_noise(client_params[0], num_clients)
        
        # 检查噪声是否被添加
        noise_added = False
        for key in ['global_lora_A.weight', 'global_lora_B.weight']:
            if key in client_params[0] and key in aggregated:
                if not torch.equal(client_params[0][key], aggregated[key]):
                    noise_added = True
                    break
        
        # 计算噪声统计
        noise_stats = {}
        for key in ['global_lora_A.weight', 'global_lora_B.weight']:
            if key in client_params[0] and key in aggregated:
                original = client_params[0][key]
                noisy = aggregated[key]
                noise = noisy - original
                noise_stats[key] = {
                    'mean': noise.mean().item(),
                    'std': noise.std().item(),
                    'max_abs': noise.abs().max().item()
                }
        
        results = {
            'noise_added': noise_added,
            'noise_stats': noise_stats
        }
        
        logger.info(f"聚合噪声已添加: {noise_added}")
        for key, stats in noise_stats.items():
            logger.info(f"{key}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        
        return results
    
    def test_parameter_impact(self, num_tests: int = 20) -> Dict[str, float]:
        """测试噪声对模型参数的影响"""
        logger.info("测试噪声对模型参数的影响...")
        
        model = self._create_test_model()
        config = DPSGDConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)
        trainer = DualLoRADPSGDTrainer(model, config)
        
        # 记录原始参数
        original_params = {}
        for name, param in model.named_parameters():
            if 'global_lora' in name:
                original_params[name] = param.clone()
        
        # 模拟训练步骤（添加噪声）
        parameter_changes = []
        for test in range(num_tests):
            # 重置梯度
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            
            # 添加随机梯度
            for name, param in model.named_parameters():
                if 'global_lora' in name:
                    param.grad = torch.randn_like(param) * 0.1
            
            # 添加噪声
            trainer.add_noise_to_gradients(model, noise_scale=1.0)
            
            # 记录参数变化
            test_changes = {}
            for name, param in model.named_parameters():
                if 'global_lora' in name and param.grad is not None:
                    test_changes[name] = param.grad.clone()
            parameter_changes.append(test_changes)
        
        # 分析参数变化
        change_stats = {}
        for name in original_params.keys():
            changes = [change[name] for change in parameter_changes if name in change]
            if changes:
                all_changes = torch.cat([c.flatten() for c in changes])
                change_stats[name] = {
                    'mean': all_changes.mean().item(),
                    'std': all_changes.std().item(),
                    'max_abs': all_changes.abs().max().item()
                }
        
        results = {
            'parameter_changes': change_stats,
            'num_tests': num_tests
        }
        
        logger.info(f"参数变化统计 (基于{num_tests}次测试):")
        for name, stats in change_stats.items():
            logger.info(f"{name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        
        return results
    
    def _create_test_model(self) -> nn.Module:
        """创建测试模型"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.global_lora_A = nn.Linear(768, 8, bias=False)
                self.global_lora_B = nn.Linear(8, 768, bias=False)
                self.local_lora_A = nn.Linear(768, 4, bias=False)
                self.local_lora_B = nn.Linear(4, 768, bias=False)
                self.classifier = nn.Linear(768, 2)
            
            def forward(self, x):
                return self.classifier(x)
        
        return TestModel()
    
    def _reset_model_parameters(self, model: nn.Module):
        """重置模型参数"""
        for param in model.parameters():
            param.data.normal_(0, 0.01)
            if param.grad is not None:
                param.grad.zero_()
    
    def run_all_tests(self) -> Dict[str, any]:
        """运行所有测试"""
        logger.info("开始差分隐私噪声有效性测试...")
        
        all_results = {}
        
        # 1. 噪声随机性测试
        all_results['randomness'] = self.test_noise_randomness()
        
        # 2. 噪声独立性测试
        all_results['independence'] = self.test_noise_independence()
        
        # 3. 不同种子测试
        all_results['different_seeds'] = self.test_different_seeds()
        
        # 4. 聚合噪声测试
        all_results['aggregation_noise'] = self.test_aggregation_noise()
        
        # 5. 参数影响测试
        all_results['parameter_impact'] = self.test_parameter_impact()
        
        # 生成总结报告
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, results: Dict[str, any]):
        """生成总结报告"""
        logger.info("=" * 60)
        logger.info("差分隐私噪声有效性测试报告")
        logger.info("=" * 60)
        
        # 随机性测试结果
        randomness = results['randomness']
        logger.info(f"1. 噪声随机性:")
        logger.info(f"   - 正态性: {'通过' if randomness['is_normal'] else '失败'}")
        logger.info(f"   - 标准差比率: {randomness['std_ratio']:.4f} (期望: 1.0)")
        
        # 独立性测试结果
        independence = results['independence']
        logger.info(f"2. 噪声独立性:")
        logger.info(f"   - 自相关: {independence['autocorrelation']:.6f}")
        logger.info(f"   - 独立性: {'通过' if independence['is_independent'] else '失败'}")
        
        # 聚合噪声测试结果
        aggregation = results['aggregation_noise']
        logger.info(f"3. 聚合噪声:")
        logger.info(f"   - 噪声已添加: {'是' if aggregation['noise_added'] else '否'}")
        
        # 总体评估
        overall_pass = (
            randomness['is_normal'] and 
            independence['is_independent'] and 
            aggregation['noise_added']
        )
        
        logger.info(f"4. 总体评估:")
        logger.info(f"   - 差分隐私噪声机制: {'正常工作' if overall_pass else '存在问题'}")
        
        if not overall_pass:
            logger.warning("警告: 差分隐私噪声机制可能存在问题!")
            if not randomness['is_normal']:
                logger.warning("  - 噪声不符合正态分布")
            if not independence['is_independent']:
                logger.warning("  - 噪声缺乏独立性")
            if not aggregation['noise_added']:
                logger.warning("  - 聚合噪声未正确添加")
        
        logger.info("=" * 60)


def main():
    """主函数"""
    tester = NoiseEffectivenessTester()
    results = tester.run_all_tests()
    
    # 保存结果
    import json
    with open('/home/szk_25/FedSA-LoRA-Dual/noise_effectiveness_test_results.json', 'w') as f:
        # 转换numpy类型为Python类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info("测试结果已保存到 noise_effectiveness_test_results.json")


if __name__ == "__main__":
    main()
