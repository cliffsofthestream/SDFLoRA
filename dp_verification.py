

import torch
import torch.nn as nn
import numpy as np
import random
import logging
from typing import Dict, List, Tuple
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入项目模块
from dp_sgd_engine import DPSGDConfig, DualLoRADPSGDTrainer
from dual_lora_aggregator import DualLoRAAggregator

class DPVerificationTester:
    """差分隐私验证测试器"""
    
    def __init__(self):
        self.results = {}
    
    def create_test_model(self):
        """创建测试模型"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.global_lora_A = nn.Linear(10, 5, bias=False)
                self.global_lora_B = nn.Linear(5, 10, bias=False)
                self.local_lora_A = nn.Linear(10, 3, bias=False)
                self.local_lora_B = nn.Linear(3, 10, bias=False)
                self.classifier = nn.Linear(10, 2)
            
            def forward(self, x):
                x = self.global_lora_A(x)
                x = self.global_lora_B(x)
                x = self.local_lora_A(x)
                x = self.local_lora_B(x)
                return self.classifier(x)
        
        return TestModel()
    
    def test_without_dp(self, seed: int = 123) -> Dict[str, float]:
        """测试不使用DP-SGD的情况"""
        logger.info(f"测试不使用DP-SGD (种子: {seed})...")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = self.create_test_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 记录初始参数
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
        
        # 模拟训练过程
        training_losses = []
        parameter_changes = []
        
        for epoch in range(5):
            epoch_loss = 0.0
            epoch_changes = {}
            
            for batch in range(3):  # 3个批次
                # 生成随机数据
                x = torch.randn(32, 10)
                y = torch.randint(0, 2, (32,))
                
                # 前向传播
                outputs = model(x)
                loss = criterion(outputs, y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 记录梯度（不添加噪声）
                grad_norms = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms[name] = param.grad.norm().item()
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 记录参数变化
            for name, param in model.named_parameters():
                change = (param - initial_params[name]).norm().item()
                epoch_changes[name] = change
            
            training_losses.append(epoch_loss / 3)
            parameter_changes.append(epoch_changes)
        
        return {
            'training_losses': training_losses,
            'parameter_changes': parameter_changes,
            'final_params': {name: param.clone() for name, param in model.named_parameters()},
            'grad_norms': grad_norms
        }
    
    def test_with_dp(self, seed: int = 123, epsilon: float = 1.0) -> Dict[str, float]:
        """测试使用DP-SGD的情况"""
        logger.info(f"测试使用DP-SGD (种子: {seed}, ε: {epsilon})...")
        
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        model = self.create_test_model()
        
        # 创建DP-SGD配置
        config = DPSGDConfig(
            epsilon=epsilon,
            delta=1e-5,
            max_grad_norm=1.0,
            apply_to_global=True,
            apply_to_local=False
        )
        
        trainer = DualLoRADPSGDTrainer(model, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 记录初始参数
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
        
        # 模拟训练过程
        training_losses = []
        parameter_changes = []
        noise_stats = []
        
        for epoch in range(5):
            epoch_loss = 0.0
            epoch_changes = {}
            epoch_noise = {}
            
            for batch in range(3):  # 3个批次
                # 生成随机数据
                x = torch.randn(32, 10)
                y = torch.randint(0, 2, (32,))
                
                # 前向传播
                outputs = model(x)
                loss = criterion(outputs, y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 记录添加噪声前的梯度
                grads_before = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grads_before[name] = param.grad.clone()
                
                # 添加DP-SGD噪声
                trainer.add_noise_to_gradients(model, noise_scale=1.0)
                
                # 记录噪声统计
                for name, param in model.named_parameters():
                    if param.grad is not None and name in grads_before:
                        noise = param.grad - grads_before[name]
                        epoch_noise[name] = {
                            'mean': noise.mean().item(),
                            'std': noise.std().item(),
                            'max_abs': noise.abs().max().item()
                        }
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # 记录参数变化
            for name, param in model.named_parameters():
                change = (param - initial_params[name]).norm().item()
                epoch_changes[name] = change
            
            training_losses.append(epoch_loss / 3)
            parameter_changes.append(epoch_changes)
            noise_stats.append(epoch_noise)
        
        return {
            'training_losses': training_losses,
            'parameter_changes': parameter_changes,
            'noise_stats': noise_stats,
            'final_params': {name: param.clone() for name, param in model.named_parameters()},
            'privacy_status': trainer.get_privacy_status()
        }
    
    def test_aggregation_noise(self, num_clients: int = 3) -> Dict[str, any]:
        """测试聚合噪声"""
        logger.info(f"测试聚合噪声 (客户端数: {num_clients})...")
        
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
        
        # 创建客户端参数
        client_params = []
        for i in range(num_clients):
            params = {
                'global_lora_A.weight': torch.randn(5, 10),
                'global_lora_B.weight': torch.randn(10, 5),
                'local_lora_A.weight': torch.randn(3, 10),
                'local_lora_B.weight': torch.randn(10, 3)
            }
            client_params.append(params)
        
        # 执行聚合（带噪声）
        aggregated = aggregator._add_aggregation_noise(client_params[0], num_clients)
        
        # 分析噪声
        noise_analysis = {}
        for key in ['global_lora_A.weight', 'global_lora_B.weight']:
            if key in client_params[0] and key in aggregated:
                original = client_params[0][key]
                noisy = aggregated[key]
                noise = noisy - original
                noise_analysis[key] = {
                    'mean': noise.mean().item(),
                    'std': noise.std().item(),
                    'max_abs': noise.abs().max().item(),
                    'noise_added': not torch.equal(original, noisy)
                }
        
        return {
            'noise_analysis': noise_analysis,
            'aggregation_successful': any(analysis['noise_added'] for analysis in noise_analysis.values())
        }
    
    def compare_experiments(self, seeds: List[int] = [123, 456, 789]) -> Dict[str, any]:
        """对比不同种子的实验结果"""
        logger.info("对比不同种子的实验结果...")
        
        results = {}
        
        for seed in seeds:
            # 不使用DP-SGD
            no_dp_result = self.test_without_dp(seed)
            
            # 使用DP-SGD
            dp_result = self.test_with_dp(seed)
            
            # 计算差异
            final_loss_diff = abs(no_dp_result['training_losses'][-1] - dp_result['training_losses'][-1])
            
            # 计算参数差异
            param_diffs = {}
            for name in no_dp_result['final_params'].keys():
                if name in dp_result['final_params']:
                    diff = (no_dp_result['final_params'][name] - dp_result['final_params'][name]).norm().item()
                    param_diffs[name] = diff
            
            results[seed] = {
                'no_dp': no_dp_result,
                'with_dp': dp_result,
                'final_loss_diff': final_loss_diff,
                'param_diffs': param_diffs,
                'dp_effective': final_loss_diff > 0.001  # 阈值判断
            }
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """运行综合测试"""
        logger.info("开始综合差分隐私验证测试...")
        
        all_results = {}
        
        # 1. 测试聚合噪声
        all_results['aggregation_test'] = self.test_aggregation_noise()
        
        # 2. 对比实验
        all_results['comparison_test'] = self.compare_experiments()
        
        # 3. 分析结果
        self._analyze_results(all_results)
        
        return all_results
    
    def _analyze_results(self, results: Dict[str, any]):
        """分析测试结果"""
        logger.info("=" * 60)
        logger.info("差分隐私验证测试结果分析")
        logger.info("=" * 60)
        
        # 聚合噪声测试结果
        agg_test = results['aggregation_test']
        logger.info(f"1. 聚合噪声测试:")
        logger.info(f"   - 聚合成功: {'是' if agg_test['aggregation_successful'] else '否'}")
        
        for key, analysis in agg_test['noise_analysis'].items():
            logger.info(f"   - {key}: 噪声已添加={'是' if analysis['noise_added'] else '否'}, "
                       f"std={analysis['std']:.6f}")
        
        # 对比实验结果
        comp_test = results['comparison_test']
        logger.info(f"2. 对比实验测试:")
        
        dp_effective_count = 0
        for seed, result in comp_test.items():
            is_effective = result['dp_effective']
            dp_effective_count += 1 if is_effective else 0
            logger.info(f"   - 种子 {seed}: DP-SGD有效={'是' if is_effective else '否'}, "
                       f"损失差异={result['final_loss_diff']:.6f}")
        
        # 总体评估
        overall_effective = (agg_test['aggregation_successful'] and 
                           dp_effective_count > 0)
        
        logger.info(f"3. 总体评估:")
        logger.info(f"   - 聚合噪声: {'✓' if agg_test['aggregation_successful'] else '✗'}")
        logger.info(f"   - DP-SGD影响: {'✓' if dp_effective_count > 0 else '✗'}")
        logger.info(f"   - 差分隐私机制: {'正常工作' if overall_effective else '存在问题'}")
        
        if not overall_effective:
            logger.warning("警告: 差分隐私机制可能未正确实现或配置!")
        else:
            logger.info("✓ 差分隐私机制正常工作!")

def main():
    """主函数"""
    tester = DPVerificationTester()
    results = tester.run_comprehensive_test()
    
    # 保存结果
    with open('/home/szk_25/FedSA-LoRA-Dual/dp_verification_results.json', 'w') as f:
        def convert_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_tensor(results), f, indent=2)
    
    logger.info("测试结果已保存到 dp_verification_results.json")

if __name__ == "__main__":
    main()
