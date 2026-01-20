

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass
import numpy as np

from dp_sgd_engine import DPSGDConfig, DualLoRADPSGDTrainer, create_dual_lora_dp_trainer
from dual_lora_adapter import DualLoRAModel, DualLoRAConfig

logger = logging.getLogger(__name__)


class DualLoRADPTrainer:
    """
    双模块LoRA DP-SGD训练器
    
    核心功能:
    1. 集成DP-SGD到双模块LoRA训练过程
    2. 分别处理全局和本地适配器的隐私保护
    3. 支持联邦学习中的隐私保护训练
    """
    
    def __init__(
        self,
        model: DualLoRAModel,
        config: DPSGDConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # 初始化DP-SGD训练器
        self.dp_trainer = DualLoRADPSGDTrainer(model, config)
        
        # 设置优化器和损失函数
        self.optimizer = optimizer or self._create_optimizer()
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # 训练统计
        self.training_stats = {
            'epoch': 0,
            'step': 0,
            'total_loss': 0.0,
            'privacy_epsilon': 0.0,
            'privacy_delta': 0.0,
            'grad_norm_global': 0.0,
            'grad_norm_local': 0.0
        }
        
        logger.info(f"Dual-LoRA DP-SGD Trainer initialized on {device}")
        logger.info(f"Privacy budget: ε={config.epsilon}, δ={config.delta}")
        logger.info(f"Apply to global: {config.apply_to_global}, Apply to local: {config.apply_to_local}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 只优化可训练参数
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        return optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
    
    def train_epoch(
        self, 
        dataloader: torch.utils.data.DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
            
        Returns:
            训练统计信息
        """
        self.model.train()
        epoch_stats = {
            'epoch': epoch,
            'total_loss': 0.0,
            'num_batches': 0,
            'privacy_epsilon': 0.0,
            'privacy_delta': 0.0,
            'avg_grad_norm_global': 0.0,
            'avg_grad_norm_local': 0.0
        }
        
        for batch_idx, batch in enumerate(dataloader):
            # 执行DP-SGD训练步骤
            step_stats = self.dp_trainer.train_step_dual_lora(
                batch, self.optimizer, self.criterion
            )
            
            # 更新统计信息
            epoch_stats['total_loss'] += step_stats['loss']
            epoch_stats['num_batches'] += 1
            epoch_stats['privacy_epsilon'] = step_stats['privacy_epsilon']
            epoch_stats['privacy_delta'] = step_stats['privacy_delta']
            epoch_stats['avg_grad_norm_global'] += step_stats.get('global_grad_norm', 0.0)
            epoch_stats['avg_grad_norm_local'] += step_stats.get('local_grad_norm', 0.0)
            
            # 更新全局统计
            self.training_stats.update({
                'epoch': epoch,
                'step': self.training_stats['step'] + 1,
                'total_loss': step_stats['loss'],
                'privacy_epsilon': step_stats['privacy_epsilon'],
                'privacy_delta': step_stats['privacy_delta'],
                'grad_norm_global': step_stats.get('global_grad_norm', 0.0),
                'grad_norm_local': step_stats.get('local_grad_norm', 0.0)
            })
            
            # 检查隐私预算
            if not self.dp_trainer.privacy_accountant.can_continue_training():
                logger.warning("Privacy budget exhausted, stopping training")
                break
        
        # 计算平均值
        if epoch_stats['num_batches'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['num_batches']
            epoch_stats['avg_grad_norm_global'] /= epoch_stats['num_batches']
            epoch_stats['avg_grad_norm_local'] /= epoch_stats['num_batches']
        
        logger.info(f"Epoch {epoch} completed: "
                   f"loss={epoch_stats['avg_loss']:.4f}, "
                   f"ε={epoch_stats['privacy_epsilon']:.4f}, "
                   f"global_grad_norm={epoch_stats['avg_grad_norm_global']:.4f}, "
                   f"local_grad_norm={epoch_stats['avg_grad_norm_local']:.4f}")
        
        return epoch_stats
    
    def train(
        self, 
        dataloader: torch.utils.data.DataLoader,
        num_epochs: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        完整训练过程
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            
        Returns:
            每个epoch的训练统计信息
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        training_history = []
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Initial privacy budget: ε={self.config.epsilon}, δ={self.config.delta}")
        
        for epoch in range(num_epochs):
            # 检查隐私预算
            if not self.dp_trainer.privacy_accountant.can_continue_training():
                logger.warning(f"Privacy budget exhausted at epoch {epoch}, stopping training")
                break
            
            # 训练一个epoch
            epoch_stats = self.train_epoch(dataloader, epoch)
            training_history.append(epoch_stats)
            
            # 打印进度
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self._print_training_progress(epoch, epoch_stats)
        
        # 打印最终统计
        self._print_final_stats(training_history)
        
        return training_history
    
    def evaluate(
        self, 
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 评估数据加载器
            
        Returns:
            评估指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, None
                
                if targets is not None:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    # 计算准确率
                    if hasattr(outputs, 'logits'):
                        predictions = torch.argmax(outputs.logits, dim=-1)
                    else:
                        predictions = torch.argmax(outputs, dim=-1)
                    
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局适配器参数（用于联邦聚合）"""
        return self.model.get_global_state_dict()
    
    def get_local_parameters(self) -> Dict[str, torch.Tensor]:
        """获取本地适配器参数（保持本地）"""
        return self.model.get_local_state_dict()
    
    def load_global_parameters(self, global_params: Dict[str, torch.Tensor]):
        """加载全局适配器参数"""
        self.model.load_global_state_dict(global_params)
    
    def get_privacy_status(self) -> Dict[str, float]:
        """获取当前隐私状态"""
        return self.dp_trainer.get_privacy_status()
    
    def _print_training_progress(self, epoch: int, stats: Dict[str, float]):
        """打印训练进度"""
        logger.info(f"Epoch {epoch:3d} | "
                   f"Loss: {stats['avg_loss']:.4f} | "
                   f"ε: {stats['privacy_epsilon']:.4f} | "
                   f"Global Grad: {stats['avg_grad_norm_global']:.4f} | "
                   f"Local Grad: {stats['avg_grad_norm_local']:.4f}")
    
    def _print_final_stats(self, training_history: List[Dict[str, float]]):
        """打印最终统计信息"""
        if not training_history:
            return
        
        final_stats = training_history[-1]
        privacy_status = self.get_privacy_status()
        
        logger.info("=" * 60)
        logger.info("Training Completed!")
        logger.info(f"Final Loss: {final_stats['avg_loss']:.4f}")
        logger.info(f"Privacy Budget Consumed: ε={privacy_status['consumed_epsilon']:.4f}")
        logger.info(f"Remaining Privacy Budget: ε={privacy_status['remaining_epsilon']:.4f}")
        logger.info(f"Total Training Steps: {self.training_stats['step']}")
        logger.info("=" * 60)


class DualLoRAFedDPTrainer(DualLoRADPTrainer):
    """
    联邦学习专用的双模块LoRA DP-SGD训练器
    """
    
    def __init__(
        self,
        model: DualLoRAModel,
        config: DPSGDConfig,
        client_id: int = 0,
        **kwargs
    ):
        super().__init__(model, config, **kwargs)
        self.client_id = client_id
        
        logger.info(f"Federated DP-SGD Trainer initialized for client {client_id}")
    
    def federated_train_round(
        self,
        dataloader: torch.utils.data.DataLoader,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        local_epochs: int = 1
    ) -> Dict[str, Any]:
        """
        执行一轮联邦学习训练
        
        Args:
            dataloader: 本地数据加载器
            global_params: 全局参数（从服务器接收）
            local_epochs: 本地训练轮数
            
        Returns:
            训练结果，包含本地参数更新
        """
        # 加载全局参数
        if global_params is not None:
            self.load_global_parameters(global_params)
        
        # 本地训练
        training_history = []
        for epoch in range(local_epochs):
            epoch_stats = self.train_epoch(dataloader, epoch)
            training_history.append(epoch_stats)
        
        # 获取参数更新
        global_params_update = self.get_global_parameters()
        local_params = self.get_local_parameters()
        privacy_status = self.get_privacy_status()
        
        # 计算参数更新（相对于初始参数）
        if global_params is not None:
            global_params_update = {
                k: global_params_update[k] - global_params[k] 
                for k in global_params_update.keys() 
                if k in global_params
            }
        
        return {
            'client_id': self.client_id,
            'global_params_update': global_params_update,
            'local_params': local_params,
            'training_history': training_history,
            'privacy_status': privacy_status,
            'num_samples': len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
        }


def create_dual_lora_dp_trainer_from_config(
    model: DualLoRAModel,
    config: Dict[str, Any],
    device: str = 'cpu'
) -> DualLoRADPTrainer:
    """
    从配置字典创建双模块LoRA DP-SGD训练器
    
    Args:
        model: 双模块LoRA模型
        config: 配置字典
        device: 设备
        
    Returns:
        DualLoRADPTrainer实例
    """
    # 提取DP-SGD配置
    dp_config = DPSGDConfig(
        epsilon=config.get('epsilon', 1.0),
        delta=config.get('delta', 1e-5),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        noise_multiplier=config.get('noise_multiplier', 1.1),
        apply_to_global=config.get('apply_to_global', True),
        apply_to_local=config.get('apply_to_local', False),
        global_noise_scale=config.get('global_noise_scale', 1.0),
        local_noise_scale=config.get('local_noise_scale', 0.5),
        batch_size=config.get('batch_size', 32),
        num_epochs=config.get('num_epochs', 10),
        learning_rate=config.get('learning_rate', 1e-4),
        federated_rounds=config.get('federated_rounds', 10),
        clients_per_round=config.get('clients_per_round', 3),
        enable_secure_aggregation=config.get('enable_secure_aggregation', True)
    )
    
    return DualLoRADPTrainer(model, dp_config, device=device)


def create_federated_dp_trainer_from_config(
    model: DualLoRAModel,
    config: Dict[str, Any],
    client_id: int = 0,
    device: str = 'cpu'
) -> DualLoRAFedDPTrainer:
    """
    从配置字典创建联邦学习DP-SGD训练器
    
    Args:
        model: 双模块LoRA模型
        config: 配置字典
        client_id: 客户端ID
        device: 设备
        
    Returns:
        DualLoRAFedDPTrainer实例
    """
    # 提取DP-SGD配置
    dp_config = DPSGDConfig(
        epsilon=config.get('epsilon', 1.0),
        delta=config.get('delta', 1e-5),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        noise_multiplier=config.get('noise_multiplier', 1.1),
        apply_to_global=config.get('apply_to_global', True),
        apply_to_local=config.get('apply_to_local', False),
        global_noise_scale=config.get('global_noise_scale', 1.0),
        local_noise_scale=config.get('local_noise_scale', 0.5),
        batch_size=config.get('batch_size', 32),
        num_epochs=config.get('num_epochs', 10),
        learning_rate=config.get('learning_rate', 1e-4),
        federated_rounds=config.get('federated_rounds', 10),
        clients_per_round=config.get('clients_per_round', 3),
        enable_secure_aggregation=config.get('enable_secure_aggregation', True)
    )
    
    return DualLoRAFedDPTrainer(model, dp_config, client_id=client_id, device=device)
