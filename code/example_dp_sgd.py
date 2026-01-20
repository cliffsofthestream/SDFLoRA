

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import yaml
import os

# 导入项目模块
from code.dual_lora_adapter import DualLoRAModel, DualLoRAConfig, create_dual_lora_model
from code.dual_lora_dp_trainer import (
    DualLoRADPTrainer, 
    DualLoRAFedDPTrainer,
    create_dual_lora_dp_trainer_from_config,
    create_federated_dp_trainer_from_config
)
from code.dual_lora_aggregator import DualLoRAFederatedAggregator
from code.dp_sgd_engine import DPSGDConfig, create_dp_sgd_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data(num_samples: int = 1000, input_dim: int = 768, num_classes: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    # 生成随机输入数据
    X = torch.randn(num_samples, input_dim)
    
    # 生成随机标签
    y = torch.randint(0, num_classes, (num_samples,))
    
    return X, y


def create_sample_model(input_dim: int = 768, hidden_dim: int = 512, num_classes: int = 2) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, num_classes)
    )


def load_config(config_path: str = "dual_lora_config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        logger.warning(f"Config file {config_path} not found, using default config")
        return {}


def example_1_standalone_dp_training():
    logger.info("=" * 60)
    logger.info("示例1: 独立DP-SGD训练")
    logger.info("=" * 60)
    
    # 创建示例数据
    X, y = create_sample_data(num_samples=1000)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建基础模型
    base_model = create_sample_model()
    
    # 创建双模块LoRA模型
    dual_lora_model = create_dual_lora_model(
        base_model=base_model,
        global_rank=8,
        local_rank=4,
        fusion_method="weighted_sum"
    )
    
    # 打印模型信息
    dual_lora_model.print_trainable_parameters()
    
    # 创建DP-SGD配置
    dp_config = create_dp_sgd_config(
        epsilon=1.0,
        delta=1e-5,
        max_grad_norm=1.0,
        apply_to_global=True,
        apply_to_local=False,
        global_noise_scale=1.0,
        local_noise_scale=0.5
    )
    
    # 创建DP-SGD训练器
    trainer = DualLoRADPTrainer(
        model=dual_lora_model,
        config=dp_config,
        device='cpu'
    )
    
    # 训练模型
    logger.info("开始训练...")
    training_history = trainer.train(dataloader, num_epochs=5)
    
    # 评估模型
    logger.info("评估模型...")
    eval_results = trainer.evaluate(dataloader)
    logger.info(f"评估结果: {eval_results}")
    
    # 打印隐私状态
    privacy_status = trainer.get_privacy_status()
    logger.info(f"隐私状态: {privacy_status}")
    
    return trainer, training_history, eval_results


def example_2_federated_dp_training():
    logger.info("=" * 60)
    logger.info("示例2: 联邦学习DP-SGD训练")
    logger.info("=" * 60)
    
    # 创建多个客户端的数据
    num_clients = 3
    client_data = []
    
    for i in range(num_clients):
        X, y = create_sample_data(num_samples=500)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        client_data.append(dataloader)
    
    # 创建基础模型
    base_model = create_sample_model()
    
    # 创建双模块LoRA模型
    dual_lora_model = create_dual_lora_model(
        base_model=base_model,
        global_rank=8,
        local_rank=4,
        fusion_method="weighted_sum"
    )
    
    # 创建DP-SGD配置
    dp_config = create_dp_sgd_config(
        epsilon=2.0,
        delta=1e-5,
        max_grad_norm=1.0,
        apply_to_global=True,
        apply_to_local=False,
        global_noise_scale=1.0,
        local_noise_scale=0.5
    )
    
    # 创建联邦聚合器
    aggregator = DualLoRAFederatedAggregator(
        model=dual_lora_model,
        enable_dp_sgd=True,
        dp_config={
            'enabled': True,
            'epsilon': 2.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
    )
    
    # 创建客户端训练器
    client_trainers = []
    for i in range(num_clients):
        trainer = DualLoRAFedDPTrainer(
            model=dual_lora_model,
            config=dp_config,
            client_id=i,
            device='cpu'
        )
        client_trainers.append(trainer)
    
    # 模拟联邦学习过程
    num_rounds = 3
    global_params = None
    
    for round_idx in range(num_rounds):
        logger.info(f"联邦学习轮次 {round_idx + 1}/{num_rounds}")
        
        # 客户端本地训练
        client_updates = []
        for i, trainer in enumerate(client_trainers):
            logger.info(f"客户端 {i} 本地训练...")
            
            # 执行本地训练
            result = trainer.federated_train_round(
                dataloader=client_data[i],
                global_params=global_params,
                local_epochs=2
            )
            
            client_updates.append(result)
            logger.info(f"客户端 {i} 隐私状态: ε={result['privacy_status']['consumed_epsilon']:.4f}")
        
        # 服务器聚合
        logger.info("服务器聚合...")
        agg_info = {
            "client_feedback": [
                (update['client_id'], (update['num_samples'], update['global_params_update']))
                for update in client_updates
            ]
        }
        
        global_params = aggregator.aggregate(agg_info)
        logger.info(f"聚合完成，全局参数数量: {len(global_params)}")
    
    # 最终评估
    logger.info("最终评估...")
    final_eval_results = []
    for i, trainer in enumerate(client_trainers):
        eval_results = trainer.evaluate(client_data[i])
        final_eval_results.append(eval_results)
        logger.info(f"客户端 {i} 最终准确率: {eval_results['accuracy']:.4f}")
    
    return client_trainers, global_params, final_eval_results


def example_3_config_based_training():
    logger.info("=" * 60)
    logger.info("示例3: 基于配置文件的训练")
    logger.info("=" * 60)
    
    # 加载配置文件
    config = load_config("dual_lora_config.yaml")
    
    if not config:
        logger.warning("无法加载配置文件，使用默认配置")
        return
    
    # 提取DP-SGD配置
    dp_config = config.get('differential_privacy', {})
    if not dp_config.get('enabled', False):
        logger.warning("配置文件中DP-SGD未启用")
        return
    
    logger.info(f"DP-SGD配置: {dp_config}")
    
    # 创建示例数据
    X, y = create_sample_data(num_samples=800)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=dp_config.get('batch_size', 32), shuffle=True)
    
    # 创建基础模型
    base_model = create_sample_model()
    
    # 创建双模块LoRA模型
    dual_lora_model = create_dual_lora_model(
        base_model=base_model,
        global_rank=8,
        local_rank=4,
        fusion_method="weighted_sum"
    )
    
    # 从配置创建训练器
    trainer = create_dual_lora_dp_trainer_from_config(
        model=dual_lora_model,
        config=dp_config,
        device='cpu'
    )
    
    # 训练模型
    logger.info("开始基于配置的训练...")
    training_history = trainer.train(dataloader, num_epochs=3)
    
    # 评估模型
    eval_results = trainer.evaluate(dataloader)
    logger.info(f"评估结果: {eval_results}")
    
    return trainer, training_history, eval_results


def example_4_privacy_analysis():
    logger.info("=" * 60)
    logger.info("示例4: 隐私分析")
    logger.info("=" * 60)
    
    # 创建不同隐私预算的配置
    privacy_configs = [
        {'epsilon': 0.1, 'delta': 1e-5, 'name': '高隐私保护'},
        {'epsilon': 1.0, 'delta': 1e-5, 'name': '中等隐私保护'},
        {'epsilon': 10.0, 'delta': 1e-5, 'name': '低隐私保护'},
    ]
    
    results = []
    
    for config in privacy_configs:
        logger.info(f"测试配置: {config['name']} (ε={config['epsilon']})")
        
        # 创建数据
        X, y = create_sample_data(num_samples=500)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 创建模型
        base_model = create_sample_model()
        dual_lora_model = create_dual_lora_model(base_model, global_rank=8, local_rank=4)
        
        # 创建训练器
        dp_config = create_dp_sgd_config(
            epsilon=config['epsilon'],
            delta=config['delta'],
            max_grad_norm=1.0
        )
        
        trainer = DualLoRADPTrainer(dual_lora_model, dp_config)
        
        # 训练
        training_history = trainer.train(dataloader, num_epochs=3)
        
        # 评估
        eval_results = trainer.evaluate(dataloader)
        privacy_status = trainer.get_privacy_status()
        
        results.append({
            'config': config,
            'accuracy': eval_results['accuracy'],
            'loss': eval_results['loss'],
            'privacy_consumed': privacy_status['consumed_epsilon'],
            'noise_multiplier': privacy_status['noise_multiplier']
        })
        
        logger.info(f"结果: 准确率={eval_results['accuracy']:.4f}, "
                   f"损失={eval_results['loss']:.4f}, "
                   f"隐私消耗={privacy_status['consumed_epsilon']:.4f}")
    
    # 打印比较结果
    logger.info("\n隐私-效用权衡分析:")
    logger.info("-" * 80)
    logger.info(f"{'配置':<20} {'准确率':<10} {'损失':<10} {'隐私消耗':<12} {'噪声乘数':<12}")
    logger.info("-" * 80)
    
    for result in results:
        config_name = result['config']['name']
        accuracy = result['accuracy']
        loss = result['loss']
        privacy_consumed = result['privacy_consumed']
        noise_multiplier = result['noise_multiplier']
        
        logger.info(f"{config_name:<20} {accuracy:<10.4f} {loss:<10.4f} "
                   f"{privacy_consumed:<12.4f} {noise_multiplier:<12.4f}")
    
    return results


def main():
    """主函数"""
    
    try:
        # 独立DP-SGD训练
        example_1_standalone_dp_training()
        
        # 联邦学习DP-SGD训练
        example_2_federated_dp_training()
        
        # 基于配置文件的训练
        example_3_config_based_training()
        
        # 隐私分析
        example_4_privacy_analysis()
        
        logger.info("所有示例执行完成！")
        
    except Exception as e:
        logger.error(f"执行过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()
