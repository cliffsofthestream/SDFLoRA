#!/usr/bin/env python3
"""
双模块LoRA DP-SGD测试脚本
验证DP-SGD功能的正确性和性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import unittest
from typing import Dict, List, Tuple, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from dp_sgd_engine import (
    DPSGDConfig, PrivacyAccountant, DPSGDTrainer, 
    DualLoRADPSGDTrainer, create_dp_sgd_config, create_dual_lora_dp_trainer
)
from dual_lora_adapter import DualLoRAModel, DualLoRAConfig, create_dual_lora_model
from dual_lora_dp_trainer import DualLoRADPTrainer, DualLoRAFedDPTrainer
from dual_lora_aggregator import DualLoRAAggregator, DualLoRAFederatedAggregator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestDPSGDEngine(unittest.TestCase):
    """测试DP-SGD引擎"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = 'cpu'
        self.input_dim = 128
        self.hidden_dim = 64
        self.num_classes = 2
        self.batch_size = 16
        self.num_samples = 100
        
        # 创建测试数据
        self.X = torch.randn(self.num_samples, self.input_dim)
        self.y = torch.randint(0, self.num_classes, (self.num_samples,))
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建测试模型
        self.base_model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
        
        # 创建双模块LoRA模型
        self.dual_lora_model = create_dual_lora_model(
            base_model=self.base_model,
            global_rank=8,
            local_rank=4,
            fusion_method="weighted_sum"
        )
    
    def test_privacy_accountant(self):
        """测试隐私预算计算器"""
        logger.info("测试隐私预算计算器...")
        
        accountant = PrivacyAccountant(epsilon=1.0, delta=1e-5)
        
        # 测试噪声乘数计算
        noise_multiplier = accountant.compute_noise_multiplier(
            target_epsilon=1.0,
            target_delta=1e-5,
            num_steps=100,
            batch_size=16,
            total_samples=1000
        )
        
        self.assertGreater(noise_multiplier, 0)
        self.assertLess(noise_multiplier, 10)  # 合理的噪声乘数范围
        
        # 测试隐私消耗计算
        epsilon_spent, delta_spent = accountant.compute_privacy_spent(
            noise_multiplier=noise_multiplier,
            num_steps=50,
            batch_size=16,
            total_samples=1000
        )
        
        self.assertGreater(epsilon_spent, 0)
        self.assertLess(epsilon_spent, 1.0)
        self.assertEqual(delta_spent, 1e-5)
        
        logger.info(f"噪声乘数: {noise_multiplier:.4f}")
        logger.info(f"隐私消耗: ε={epsilon_spent:.4f}, δ={delta_spent:.4f}")
    
    def test_dp_sgd_config(self):
        """测试DP-SGD配置"""
        logger.info("测试DP-SGD配置...")
        
        config = create_dp_sgd_config(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            apply_to_global=True,
            apply_to_local=False
        )
        
        self.assertEqual(config.epsilon, 1.0)
        self.assertEqual(config.delta, 1e-5)
        self.assertEqual(config.max_grad_norm, 1.0)
        self.assertTrue(config.apply_to_global)
        self.assertFalse(config.apply_to_local)
        
        logger.info("DP-SGD配置测试通过")
    
    def test_dp_sgd_trainer(self):
        """测试DP-SGD训练器"""
        logger.info("测试DP-SGD训练器...")
        
        config = create_dp_sgd_config(epsilon=1.0, delta=1e-5)
        trainer = DPSGDTrainer(self.base_model, config)
        
        # 测试梯度裁剪
        for param in self.base_model.parameters():
            param.grad = torch.randn_like(param) * 10  # 大梯度
        
        grad_norm = trainer.clip_gradients(self.base_model, max_norm=1.0)
        self.assertLessEqual(grad_norm, 1.0 + 1e-6)  # 允许小的数值误差
        
        # 测试噪声添加
        original_grads = {}
        for name, param in self.base_model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()
        
        trainer.add_noise_to_gradients(self.base_model)
        
        # 检查梯度是否被修改
        for name, param in self.base_model.named_parameters():
            if param.grad is not None and name in original_grads:
                self.assertFalse(torch.equal(param.grad, original_grads[name]))
        
        logger.info("DP-SGD训练器测试通过")
    
    def test_dual_lora_dp_trainer(self):
        """测试双模块LoRA DP-SGD训练器"""
        logger.info("测试双模块LoRA DP-SGD训练器...")
        
        config = create_dp_sgd_config(
            epsilon=1.0,
            delta=1e-5,
            apply_to_global=True,
            apply_to_local=False
        )
        
        trainer = DualLoRADPSGDTrainer(self.dual_lora_model, config)
        
        # 测试参数分离
        global_params = trainer._extract_global_parameters()
        local_params = trainer._extract_local_parameters()
        
        self.assertGreater(len(global_params), 0)
        self.assertGreater(len(local_params), 0)
        
        # 测试训练步骤
        batch = next(iter(self.dataloader))
        optimizer = optim.Adam(self.dual_lora_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        step_stats = trainer.train_step_dual_lora(batch, optimizer, criterion)
        
        self.assertIn('loss', step_stats)
        self.assertIn('global_grad_norm', step_stats)
        self.assertIn('local_grad_norm', step_stats)
        self.assertIn('privacy_epsilon', step_stats)
        
        logger.info(f"训练步骤统计: {step_stats}")
        logger.info("双模块LoRA DP-SGD训练器测试通过")
    
    def test_privacy_budget_management(self):
        """测试隐私预算管理"""
        logger.info("测试隐私预算管理...")
        
        config = create_dp_sgd_config(epsilon=0.5, delta=1e-5)
        trainer = DualLoRADPSGDTrainer(self.dual_lora_model, config)
        
        # 模拟多次训练步骤
        optimizer = optim.Adam(self.dual_lora_model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        initial_epsilon = trainer.privacy_accountant.consumed_epsilon
        
        for i in range(5):
            batch = next(iter(self.dataloader))
            step_stats = trainer.train_step_dual_lora(batch, optimizer, criterion)
            
            current_epsilon = step_stats['privacy_epsilon']
            self.assertGreaterEqual(current_epsilon, initial_epsilon)
            initial_epsilon = current_epsilon
        
        # 检查隐私预算是否被正确消耗
        final_status = trainer.get_privacy_status()
        self.assertGreater(final_status['consumed_epsilon'], 0)
        self.assertLess(final_status['consumed_epsilon'], config.epsilon)
        
        logger.info(f"最终隐私状态: {final_status}")
        logger.info("隐私预算管理测试通过")


class TestDualLoRADPTrainer(unittest.TestCase):
    """测试双模块LoRA DP训练器"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = 'cpu'
        self.input_dim = 128
        self.num_classes = 2
        self.num_samples = 200
        
        # 创建测试数据
        self.X = torch.randn(self.num_samples, self.input_dim)
        self.y = torch.randint(0, self.num_classes, (self.num_samples,))
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        # 创建测试模型
        self.base_model = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )
        
        self.dual_lora_model = create_dual_lora_model(
            base_model=self.base_model,
            global_rank=8,
            local_rank=4
        )
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        logger.info("测试训练器初始化...")
        
        config = create_dp_sgd_config(epsilon=1.0, delta=1e-5)
        trainer = DualLoRADPTrainer(self.dual_lora_model, config)
        
        self.assertIsNotNone(trainer.dp_trainer)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
        
        logger.info("训练器初始化测试通过")
    
    def test_training_epoch(self):
        """测试训练epoch"""
        logger.info("测试训练epoch...")
        
        config = create_dp_sgd_config(epsilon=1.0, delta=1e-5)
        trainer = DualLoRADPTrainer(self.dual_lora_model, config)
        
        # 训练一个epoch
        epoch_stats = trainer.train_epoch(self.dataloader, epoch=0)
        
        self.assertIn('epoch', epoch_stats)
        self.assertIn('avg_loss', epoch_stats)
        self.assertIn('privacy_epsilon', epoch_stats)
        self.assertIn('avg_grad_norm_global', epoch_stats)
        self.assertIn('avg_grad_norm_local', epoch_stats)
        
        logger.info(f"Epoch统计: {epoch_stats}")
        logger.info("训练epoch测试通过")
    
    def test_model_evaluation(self):
        """测试模型评估"""
        logger.info("测试模型评估...")
        
        config = create_dp_sgd_config(epsilon=1.0, delta=1e-5)
        trainer = DualLoRADPTrainer(self.dual_lora_model, config)
        
        # 评估模型
        eval_results = trainer.evaluate(self.dataloader)
        
        self.assertIn('accuracy', eval_results)
        self.assertIn('loss', eval_results)
        self.assertIn('correct', eval_results)
        self.assertIn('total', eval_results)
        
        self.assertGreaterEqual(eval_results['accuracy'], 0.0)
        self.assertLessEqual(eval_results['accuracy'], 1.0)
        
        logger.info(f"评估结果: {eval_results}")
        logger.info("模型评估测试通过")
    
    def test_parameter_management(self):
        """测试参数管理"""
        logger.info("测试参数管理...")
        
        config = create_dp_sgd_config(epsilon=1.0, delta=1e-5)
        trainer = DualLoRADPTrainer(self.dual_lora_model, config)
        
        # 获取全局参数
        global_params = trainer.get_global_parameters()
        self.assertGreater(len(global_params), 0)
        
        # 获取本地参数
        local_params = trainer.get_local_parameters()
        self.assertGreater(len(local_params), 0)
        
        # 测试参数加载
        original_global = global_params.copy()
        trainer.load_global_parameters(global_params)
        
        # 验证参数是否被正确加载
        new_global = trainer.get_global_parameters()
        for key in original_global:
            self.assertTrue(torch.equal(original_global[key], new_global[key]))
        
        logger.info("参数管理测试通过")


class TestFederatedDPAggregator(unittest.TestCase):
    """测试联邦学习DP聚合器"""
    
    def setUp(self):
        """设置测试环境"""
        self.device = 'cpu'
        self.num_clients = 3
        
        # 创建测试模型
        self.base_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.dual_lora_model = create_dual_lora_model(
            base_model=self.base_model,
            global_rank=4,
            local_rank=2
        )
        
        # 创建模拟的客户端参数
        self.client_models = []
        for i in range(self.num_clients):
            model_state = {}
            for name, param in self.dual_lora_model.named_parameters():
                if 'global_lora_A' in name or 'global_lora_B' in name:
                    model_state[name] = torch.randn_like(param) + i * 0.1
                else:
                    model_state[name] = param.clone()
            self.client_models.append(model_state)
    
    def test_aggregator_initialization(self):
        """测试聚合器初始化"""
        logger.info("测试聚合器初始化...")
        
        dp_config = {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
        
        aggregator = DualLoRAAggregator(
            model=self.dual_lora_model,
            enable_dp_sgd=True,
            dp_config=dp_config
        )
        
        self.assertTrue(aggregator.enable_dp_sgd)
        self.assertEqual(aggregator.dp_config['epsilon'], 1.0)
        
        logger.info("聚合器初始化测试通过")
    
    def test_private_aggregation(self):
        """测试隐私保护聚合"""
        logger.info("测试隐私保护聚合...")
        
        dp_config = {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
        
        aggregator = DualLoRAAggregator(
            model=self.dual_lora_model,
            enable_dp_sgd=True,
            dp_config=dp_config
        )
        
        # 准备聚合信息
        agg_info = {
            "client_feedback": [
                (i, (100, model_state)) for i, model_state in enumerate(self.client_models)
            ]
        }
        
        # 执行聚合
        aggregated_params = aggregator.aggregate(agg_info)
        
        self.assertGreater(len(aggregated_params), 0)
        
        # 检查是否包含全局参数
        global_param_count = sum(1 for key in aggregated_params.keys() 
                               if 'global_lora_A' in key or 'global_lora_B' in key)
        self.assertGreater(global_param_count, 0)
        
        logger.info(f"聚合参数数量: {len(aggregated_params)}")
        logger.info(f"全局参数数量: {global_param_count}")
        logger.info("隐私保护聚合测试通过")
    
    def test_noise_addition(self):
        """测试噪声添加"""
        logger.info("测试噪声添加...")
        
        dp_config = {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
        
        aggregator = DualLoRAAggregator(
            model=self.dual_lora_model,
            enable_dp_sgd=True,
            dp_config=dp_config
        )
        
        # 创建测试参数
        test_params = {
            'global_lora_A.weight': torch.randn(4, 64),
            'global_lora_B.weight': torch.randn(64, 4),
            'local_lora_A.weight': torch.randn(2, 64),
            'local_lora_B.weight': torch.randn(64, 2)
        }
        
        # 添加噪声
        noisy_params = aggregator._add_aggregation_noise(test_params, num_clients=3)
        
        # 检查全局参数是否被添加噪声
        for key in ['global_lora_A.weight', 'global_lora_B.weight']:
            if key in test_params and key in noisy_params:
                self.assertFalse(torch.equal(test_params[key], noisy_params[key]))
        
        # 检查本地参数是否未被添加噪声
        for key in ['local_lora_A.weight', 'local_lora_B.weight']:
            if key in test_params and key in noisy_params:
                self.assertTrue(torch.equal(test_params[key], noisy_params[key]))
        
        logger.info("噪声添加测试通过")


def run_performance_test():
    """运行性能测试"""
    logger.info("=" * 60)
    logger.info("性能测试")
    logger.info("=" * 60)
    
    # 创建较大的测试数据
    input_dim = 512
    hidden_dim = 256
    num_classes = 10
    num_samples = 1000
    batch_size = 32
    
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    base_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_classes)
    )
    
    dual_lora_model = create_dual_lora_model(
        base_model=base_model,
        global_rank=16,
        local_rank=8
    )
    
    # 测试不同隐私预算的性能
    privacy_configs = [
        {'epsilon': 0.1, 'name': '高隐私'},
        {'epsilon': 1.0, 'name': '中等隐私'},
        {'epsilon': 10.0, 'name': '低隐私'},
    ]
    
    results = []
    
    for config in privacy_configs:
        logger.info(f"测试配置: {config['name']} (ε={config['epsilon']})")
        
        # 创建训练器
        dp_config = create_dp_sgd_config(
            epsilon=config['epsilon'],
            delta=1e-5,
            max_grad_norm=1.0
        )
        
        trainer = DualLoRADPTrainer(dual_lora_model, dp_config)
        
        # 训练
        import time
        start_time = time.time()
        training_history = trainer.train(dataloader, num_epochs=2)
        training_time = time.time() - start_time
        
        # 评估
        eval_results = trainer.evaluate(dataloader)
        privacy_status = trainer.get_privacy_status()
        
        results.append({
            'config': config,
            'accuracy': eval_results['accuracy'],
            'loss': eval_results['loss'],
            'training_time': training_time,
            'privacy_consumed': privacy_status['consumed_epsilon'],
            'noise_multiplier': privacy_status['noise_multiplier']
        })
        
        logger.info(f"结果: 准确率={eval_results['accuracy']:.4f}, "
                   f"训练时间={training_time:.2f}s, "
                   f"隐私消耗={privacy_status['consumed_epsilon']:.4f}")
    
    # 打印性能比较
    logger.info("\n性能比较:")
    logger.info("-" * 100)
    logger.info(f"{'配置':<15} {'准确率':<10} {'损失':<10} {'训练时间(s)':<12} {'隐私消耗':<12} {'噪声乘数':<12}")
    logger.info("-" * 100)
    
    for result in results:
        config_name = result['config']['name']
        accuracy = result['accuracy']
        loss = result['loss']
        training_time = result['training_time']
        privacy_consumed = result['privacy_consumed']
        noise_multiplier = result['noise_multiplier']
        
        logger.info(f"{config_name:<15} {accuracy:<10.4f} {loss:<10.4f} "
                   f"{training_time:<12.2f} {privacy_consumed:<12.4f} {noise_multiplier:<12.4f}")
    
    return results


def main():
    """主函数"""
    logger.info("开始DP-SGD功能测试")
    
    # 运行单元测试
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.makeSuite(TestDPSGDEngine))
    test_suite.addTest(unittest.makeSuite(TestDualLoRADPTrainer))
    test_suite.addTest(unittest.makeSuite(TestFederatedDPAggregator))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # 运行性能测试
    if test_result.wasSuccessful():
        logger.info("所有单元测试通过，开始性能测试...")
        performance_results = run_performance_test()
        
        logger.info("=" * 60)
        logger.info("测试完成总结")
        logger.info("=" * 60)
        logger.info(f"单元测试: {'通过' if test_result.wasSuccessful() else '失败'}")
        logger.info(f"性能测试: 完成 {len(performance_results)} 个配置")
        logger.info("DP-SGD功能测试完成！")
    else:
        logger.error("单元测试失败，跳过性能测试")
        logger.error(f"失败数量: {len(test_result.failures)}")
        logger.error(f"错误数量: {len(test_result.errors)}")


if __name__ == "__main__":
    main()
