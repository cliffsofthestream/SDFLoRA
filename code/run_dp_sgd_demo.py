#!/usr/bin/env python3
"""

set up:
python run_dp_sgd_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data():
    """创建演示数据"""
    logger.info("创建演示数据...")
    
    # 生成随机数据
    X = torch.randn(200, 128)  # 200个样本，128维特征
    y = torch.randint(0, 2, (200,))  # 二分类标签
    
    # 创建数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    logger.info(f"数据创建完成: {len(dataset)} 个样本")
    return dataloader

def create_demo_model():
    """创建演示模型"""
    logger.info("创建演示模型...")
    
    # 创建基础模型
    base_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 2)
    )
    
    # 创建双模块LoRA模型
    try:
        from code.dual_lora_adapter import create_dual_lora_model
        dual_lora_model = create_dual_lora_model(
            base_model=base_model,
            global_rank=8,
            local_rank=4,
            fusion_method="weighted_sum"
        )
        logger.info("双模块LoRA模型创建成功")
        return dual_lora_model
    except ImportError as e:
        logger.error(f"导入双模块LoRA模块失败: {e}")
        logger.info("使用基础模型进行演示...")
        return base_model

def demo_dp_sgd_training():
    """演示DP-SGD训练"""
    logger.info("=" * 60)
    logger.info("DP-SGD训练演示")
    logger.info("=" * 60)
    
    try:
        # 导入DP-SGD模块
        from code.dp_sgd_engine import create_dp_sgd_config, create_dual_lora_dp_trainer
        from code.dual_lora_dp_trainer import DualLoRADPTrainer
        
        # 创建数据和模型
        dataloader = create_demo_data()
        model = create_demo_model()
        
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
        
        logger.info(f"DP-SGD配置: ε={dp_config.epsilon}, δ={dp_config.delta}")
        
        # 创建DP-SGD训练器
        trainer = DualLoRADPTrainer(model, dp_config)
        
        # 训练模型
        logger.info("开始DP-SGD训练...")
        training_history = trainer.train(dataloader, num_epochs=3)
        
        # 评估模型
        logger.info("评估模型...")
        eval_results = trainer.evaluate(dataloader)
        
        # 打印结果
        logger.info("=" * 40)
        logger.info("训练结果:")
        logger.info(f"最终准确率: {eval_results['accuracy']:.4f}")
        logger.info(f"最终损失: {eval_results['loss']:.4f}")
        
        # 打印隐私状态
        privacy_status = trainer.get_privacy_status()
        logger.info("隐私状态:")
        logger.info(f"  消耗的隐私预算: ε={privacy_status['consumed_epsilon']:.4f}")
        logger.info(f"  剩余隐私预算: ε={privacy_status['remaining_epsilon']:.4f}")
        logger.info(f"  噪声乘数: {privacy_status['noise_multiplier']:.4f}")
        
        return True
        
    except ImportError as e:
        logger.error(f"导入DP-SGD模块失败: {e}")
        logger.info("请确保所有依赖模块都已正确安装")
        return False
    except Exception as e:
        logger.error(f"DP-SGD训练演示失败: {e}")
        return False

def demo_privacy_analysis():
    """演示隐私分析"""
    logger.info("=" * 60)
    logger.info("隐私分析演示")
    logger.info("=" * 60)
    
    try:
        from code.dp_sgd_engine import create_dp_sgd_config, PrivacyAccountant
        
        # 测试不同隐私预算的配置
        privacy_configs = [
            {'epsilon': 0.1, 'name': '高隐私保护'},
            {'epsilon': 1.0, 'name': '中等隐私保护'},
            {'epsilon': 10.0, 'name': '低隐私保护'},
        ]
        
        logger.info("隐私预算分析:")
        logger.info("-" * 60)
        logger.info(f"{'配置':<15} {'ε':<8} {'噪声乘数':<12} {'隐私强度':<12}")
        logger.info("-" * 60)
        
        for config in privacy_configs:
            # 创建隐私计算器
            accountant = PrivacyAccountant(config['epsilon'], 1e-5)
            
            # 计算噪声乘数
            noise_multiplier = accountant.compute_noise_multiplier(
                target_epsilon=config['epsilon'],
                target_delta=1e-5,
                num_steps=100,
                batch_size=32,
                total_samples=1000
            )
            
            # 计算隐私强度（噪声乘数的倒数）
            privacy_strength = 1.0 / noise_multiplier if noise_multiplier > 0 else float('inf')
            
            logger.info(f"{config['name']:<15} {config['epsilon']:<8.1f} "
                       f"{noise_multiplier:<12.4f} {privacy_strength:<12.4f}")
        
        logger.info("-" * 60)
        logger.info("说明: 噪声乘数越小，隐私保护越强")
        
        return True
        
    except ImportError as e:
        logger.error(f"导入隐私分析模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"隐私分析演示失败: {e}")
        return False

def demo_federated_aggregation():
    """演示联邦聚合"""
    logger.info("=" * 60)
    logger.info("联邦聚合演示")
    logger.info("=" * 60)
    
    try:
        from code.dual_lora_aggregator import DualLoRAAggregator
        
        # 创建模型
        model = create_demo_model()
        
        # 创建带DP-SGD的聚合器
        dp_config = {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
            'max_grad_norm': 1.0,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        }
        
        aggregator = DualLoRAAggregator(
            model=model,
            enable_dp_sgd=True,
            dp_config=dp_config
        )
        
        # 创建模拟的客户端参数
        client_models = []
        for i in range(3):
            model_state = {}
            for name, param in model.named_parameters():
                if 'global_lora_A' in name or 'global_lora_B' in name:
                    model_state[name] = torch.randn_like(param) + i * 0.1
                else:
                    model_state[name] = param.clone()
            client_models.append(model_state)
        
        # 准备聚合信息
        agg_info = {
            "client_feedback": [
                (i, (100, model_state)) for i, model_state in enumerate(client_models)
            ]
        }
        
        # 执行聚合
        logger.info("执行带隐私保护的联邦聚合...")
        aggregated_params = aggregator.aggregate(agg_info)
        
        logger.info(f"聚合完成: {len(aggregated_params)} 个参数")
        
        # 检查全局参数
        global_param_count = sum(1 for key in aggregated_params.keys() 
                               if 'global_lora_A' in key or 'global_lora_B' in key)
        logger.info(f"全局参数数量: {global_param_count}")
        
        return True
        
    except ImportError as e:
        logger.error(f"导入联邦聚合模块失败: {e}")
        return False
    except Exception as e:
        logger.error(f"联邦聚合演示失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("FedSA-LoRA-Dual DP-SGD演示程序")
    logger.info("参考: IMPROVING LORA IN PRIVACY-PRESERVING FEDERATED LEARNING")
    logger.info("=" * 80)
    
    # 检查依赖
    try:
        import torch
        import numpy as np
        logger.info("✓ PyTorch和NumPy已安装")
    except ImportError as e:
        logger.error(f"✗ 缺少依赖: {e}")
        return
    
    # 运行演示
    demos = [
        ("DP-SGD训练演示", demo_dp_sgd_training),
        ("隐私分析演示", demo_privacy_analysis),
        ("联邦聚合演示", demo_federated_aggregation),
    ]
    
    success_count = 0
    total_count = len(demos)
    
    for demo_name, demo_func in demos:
        logger.info(f"\n开始 {demo_name}...")
        try:
            if demo_func():
                logger.info(f"✓ {demo_name} 完成")
                success_count += 1
            else:
                logger.error(f"✗ {demo_name} 失败")
        except Exception as e:
            logger.error(f"✗ {demo_name} 出错: {e}")
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("演示总结:")
    logger.info(f"成功: {success_count}/{total_count}")
    
    if success_count == total_count:
        logger.info("🎉 所有演示都成功完成！")
        logger.info("\n接下来你可以:")
        logger.info("1. 运行完整示例: python example_dp_sgd.py")
        logger.info("2. 运行测试: python test_dp_sgd.py")
        logger.info("3. 查看配置: cat dual_lora_config.yaml")
        logger.info("4. 阅读文档: README.md")
    else:
        logger.warning("⚠️ 部分演示失败，请检查依赖和配置")
        logger.info("\n故障排除:")
        logger.info("1. 确保所有Python文件都在同一目录")
        logger.info("2. 检查PyTorch版本兼容性")
        logger.info("3. 查看错误日志获取详细信息")

if __name__ == "__main__":
    main()
