"""
差分隐私随机梯度下降引擎
为FedSA-LoRA-Dual项目提供差分隐私保护

1. 梯度裁剪和噪声添加
2. 隐私预算管理
3. 双模块LoRA的差分隐私适配
4. 联邦学习中的隐私保护聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPSGDConfig:
    """DP-SGD配置类"""
    # 隐私参数
    epsilon: float = 1.0          # 隐私预算
    delta: float = 1e-5          # 失败概率
    noise_multiplier: float = 1.1  # 噪声乘数
    
    # 梯度裁剪
    max_grad_norm: float = 1.0   # 梯度裁剪阈值
    
    # 训练参数
    batch_size: int = 32         # 批次大小
    num_epochs: int = 10         # 训练轮数
    learning_rate: float = 1e-4  # 学习率
    
    # 双模块LoRA特定配置
    apply_to_global: bool = True   # 是否对全局适配器应用DP-SGD
    apply_to_local: bool = False   # 是否对本地适配器应用DP-SGD
    global_noise_scale: float = 1.0  # 全局适配器噪声缩放
    local_noise_scale: float = 0.5   # 本地适配器噪声缩放
    
    # 联邦学习配置
    federated_rounds: int = 10    # 联邦学习轮数
    clients_per_round: int = 3    # 每轮参与的客户端数
    enable_secure_aggregation: bool = True  # 是否启用安全聚合


class PrivacyAccountant:
    """隐私预算计算器"""
    
    def __init__(self, epsilon: float, delta: float):
        self.epsilon = epsilon
        self.delta = delta
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
    
    def compute_noise_multiplier(
        self, 
        target_epsilon: float, 
        target_delta: float, 
        num_steps: int,
        batch_size: int,
        total_samples: int
    ) -> float:
        """
        计算噪声乘数
        
        Args:
            target_epsilon: 目标隐私预算
            target_delta: 目标失败概率
            num_steps: 训练步数
            batch_size: 批次大小
            total_samples: 总样本数
            
        Returns:
            噪声乘数
        """
        # 使用RDP (Renyi Differential Privacy) 计算
        q = batch_size / total_samples  # 采样概率
        
        # 计算RDP参数
        alpha = 1 + 1 / (2 * math.log(1 / target_delta))
        
        # 计算噪声乘数
        sigma_squared = (2 * alpha * q * q * num_steps) / (target_epsilon * target_epsilon)
        noise_multiplier = math.sqrt(sigma_squared)
        
        return noise_multiplier
    
    def compute_privacy_spent(
        self, 
        noise_multiplier: float, 
        num_steps: int,
        batch_size: int,
        total_samples: int
    ) -> Tuple[float, float]:
        """
        计算已消耗的隐私预算
        
        Returns:
            (epsilon_spent, delta_spent)
        """
        q = batch_size / total_samples
        
        # 使用RDP计算
        alpha = 1 + 1 / (2 * math.log(1 / self.delta))
        
        # 计算epsilon
        epsilon_spent = (alpha - 1) * q * q * num_steps / (2 * noise_multiplier * noise_multiplier)
        
        return epsilon_spent, self.delta
    
    def can_continue_training(self, additional_epsilon: float = 0.1) -> bool:
        """检查是否可以继续训练"""
        return (self.consumed_epsilon + additional_epsilon) <= self.epsilon


class DPSGDTrainer:
    """DP-SGD训练器"""
    
    def __init__(self, model: nn.Module, config: DPSGDConfig):
        self.model = model
        self.config = config
        self.privacy_accountant = PrivacyAccountant(config.epsilon, config.delta)
        
        # 计算噪声乘数
        self.noise_multiplier = self.privacy_accountant.compute_noise_multiplier(
            config.epsilon, config.delta, 
            config.num_epochs, config.batch_size, 
            config.batch_size * 100  # 假设总样本数
        )
        
        logger.info(f"DP-SGD initialized with noise_multiplier={self.noise_multiplier:.4f}")
    
    def clip_gradients(self, model: nn.Module, max_norm: float = None) -> float:
        """
        梯度裁剪
        
        Args:
            model: 模型
            max_norm: 最大梯度范数
            
        Returns:
            实际裁剪的梯度范数
        """
        if max_norm is None:
            max_norm = self.config.max_grad_norm
        
        # 计算所有参数的梯度范数
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # 裁剪梯度
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise_to_gradients(
        self, 
        model: nn.Module, 
        noise_scale: float = 1.0,
        parameter_filter: Optional[callable] = None
    ):
        """
        向梯度添加噪声
        
        Args:
            model: 模型
            noise_scale: 噪声缩放因子
            parameter_filter: 参数过滤器函数，用于选择要添加噪声的参数
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 应用参数过滤器
                if parameter_filter is not None and not parameter_filter(name):
                    continue
                
                # 计算噪声
                noise_std = self.noise_multiplier * noise_scale * self.config.max_grad_norm
                noise = torch.normal(0, noise_std, param.grad.shape, device=param.device)
                
                # 添加噪声到梯度
                param.grad.data.add_(noise)
    
    def train_step(
        self, 
        model: nn.Module, 
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        is_dual_lora: bool = False
    ) -> Dict[str, float]:
        """
        执行一个DP-SGD训练步骤
        
        Args:
            model: 模型
            batch: 批次数据
            optimizer: 优化器
            criterion: 损失函数
            is_dual_lora: 是否为双模块LoRA模型
            
        Returns:
            训练统计信息
        """
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch, None
        
        if targets is not None:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        grad_norm = self.clip_gradients(model, self.config.max_grad_norm)
        
        # 添加噪声
        if is_dual_lora:
            # 双模块LoRA：分别处理全局和本地参数
            self.add_noise_to_gradients(
                model, 
                noise_scale=self.config.global_noise_scale,
                parameter_filter=lambda name: self._is_global_parameter(name)
            )
            
            if self.config.apply_to_local:
                self.add_noise_to_gradients(
                    model,
                    noise_scale=self.config.local_noise_scale,
                    parameter_filter=lambda name: self._is_local_parameter(name)
                )
        else:
            # 标准模型：对所有参数添加噪声
            self.add_noise_to_gradients(model)
        
        # 更新参数
        optimizer.step()
        
        # 更新隐私预算
        self.privacy_accountant.consumed_epsilon += self._compute_step_privacy()
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'privacy_epsilon': self.privacy_accountant.consumed_epsilon,
            'privacy_delta': self.privacy_accountant.consumed_delta
        }
    
    def _is_global_parameter(self, param_name: str) -> bool:
        """判断是否为全局适配器参数"""
        return (
            "global_lora_A" in param_name or 
            "global_lora_B" in param_name
        )
    
    def _is_local_parameter(self, param_name: str) -> bool:
        """判断是否为本地适配器参数"""
        return (
            "local_lora_A" in param_name or 
            "local_lora_B" in param_name or
            "global_weight" in param_name or
            "local_weight" in param_name or
            "attention" in param_name or
            "gate" in param_name
        )
    
    def _compute_step_privacy(self) -> float:
        """计算单步的隐私消耗"""
        # 简化计算，实际应该使用更精确的RDP计算
        return self.config.epsilon / (self.config.num_epochs * 10)  # 假设每轮10步
    
    def get_privacy_status(self) -> Dict[str, float]:
        """获取当前隐私状态"""
        return {
            'consumed_epsilon': self.privacy_accountant.consumed_epsilon,
            'consumed_delta': self.privacy_accountant.consumed_delta,
            'remaining_epsilon': self.config.epsilon - self.privacy_accountant.consumed_epsilon,
            'noise_multiplier': self.noise_multiplier
        }


class DualLoRADPSGDTrainer(DPSGDTrainer):
    """双模块LoRA专用的DP-SGD训练器"""
    
    def __init__(self, model: nn.Module, config: DPSGDConfig):
        super().__init__(model, config)
        
        # 双模块LoRA特定配置
        self.global_parameters = self._extract_global_parameters()
        self.local_parameters = self._extract_local_parameters()
        
        logger.info(f"Dual-LoRA DP-SGD initialized:")
        logger.info(f"  Global parameters: {len(self.global_parameters)}")
        logger.info(f"  Local parameters: {len(self.local_parameters)}")
    
    def _extract_global_parameters(self) -> List[str]:
        """提取全局适配器参数名称"""
        global_params = []
        for name, param in self.model.named_parameters():
            if self._is_global_parameter(name):
                global_params.append(name)
        return global_params
    
    def _extract_local_parameters(self) -> List[str]:
        """提取本地适配器参数名称"""
        local_params = []
        for name, param in self.model.named_parameters():
            if self._is_local_parameter(name):
                local_params.append(name)
        return local_params
    
    def train_step_dual_lora(
        self, 
        batch: Tuple[torch.Tensor, ...],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """
        双模块LoRA的DP-SGD训练步骤
        
        分别对全局和本地适配器应用不同的隐私保护策略
        """
        model = self.model
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            inputs, targets = batch, None
        
        if targets is not None:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0)
        
        # 反向传播
        loss.backward()
        
        # 分别处理全局和本地参数的梯度
        global_grad_norm = self._clip_and_noise_global_gradients()
        local_grad_norm = self._clip_and_noise_local_gradients()
        
        # 更新参数
        optimizer.step()
        
        # 更新隐私预算
        self.privacy_accountant.consumed_epsilon += self._compute_step_privacy()
        
        return {
            'loss': loss.item(),
            'global_grad_norm': global_grad_norm,
            'local_grad_norm': local_grad_norm,
            'privacy_epsilon': self.privacy_accountant.consumed_epsilon,
            'privacy_delta': self.privacy_accountant.consumed_delta
        }
    
    def _clip_and_noise_global_gradients(self) -> float:
        """裁剪和添加噪声到全局适配器梯度"""
        if not self.config.apply_to_global:
            return 0.0
        
        # 计算全局参数梯度范数
        global_norm = 0.0
        for name in self.global_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                global_norm += param_norm.item() ** 2
        global_norm = global_norm ** (1. / 2)
        
        # 裁剪全局梯度
        clip_coef = min(1.0, self.config.max_grad_norm / (global_norm + 1e-6))
        for name in self.global_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
        
        # 添加噪声到全局梯度
        for name in self.global_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                noise_std = (self.noise_multiplier * self.config.global_noise_scale * 
                           self.config.max_grad_norm)
                noise = torch.normal(0, noise_std, param.grad.shape, device=param.device)
                param.grad.data.add_(noise)
        
        return global_norm
    
    def _clip_and_noise_local_gradients(self) -> float:
        """裁剪和添加噪声到本地适配器梯度"""
        if not self.config.apply_to_local:
            return 0.0
        
        # 计算本地参数梯度范数
        local_norm = 0.0
        for name in self.local_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                local_norm += param_norm.item() ** 2
        local_norm = local_norm ** (1. / 2)
        
        # 裁剪本地梯度
        clip_coef = min(1.0, self.config.max_grad_norm / (local_norm + 1e-6))
        for name in self.local_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
        
        # 添加噪声到本地梯度
        for name in self.local_parameters:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                noise_std = (self.noise_multiplier * self.config.local_noise_scale * 
                           self.config.max_grad_norm)
                noise = torch.normal(0, noise_std, param.grad.shape, device=param.device)
                param.grad.data.add_(noise)
        
        return local_norm


class DPSGDAggregator:
    """差分隐私聚合器"""
    
    def __init__(self, config: DPSGDConfig):
        self.config = config
        self.privacy_accountant = PrivacyAccountant(config.epsilon, config.delta)
    
    def aggregate_with_privacy(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        noise_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        带隐私保护的模型聚合
        
        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表
            noise_scale: 噪声缩放因子
            
        Returns:
            聚合后的模型参数
        """
        if not client_models:
            return {}
        
        # 加权平均聚合
        aggregated = {}
        template = client_models[0]
        
        for key in template.keys():
            # 计算加权平均
            weighted_sum = torch.zeros_like(template[key])
            for model, weight in zip(client_models, client_weights):
                if key in model:
                    weighted_sum += model[key] * weight
            
            # 添加聚合噪声
            if self.config.enable_secure_aggregation:
                noise_std = (self.noise_multiplier * noise_scale * 
                           self.config.max_grad_norm / len(client_models))
                noise = torch.normal(0, noise_std, weighted_sum.shape, device=weighted_sum.device)
                weighted_sum += noise
            
            aggregated[key] = weighted_sum
        
        # 更新隐私预算
        self.privacy_accountant.consumed_epsilon += self._compute_aggregation_privacy()
        
        return aggregated
    
    def _compute_aggregation_privacy(self) -> float:
        """计算聚合步骤的隐私消耗"""
        # 聚合步骤的隐私消耗通常较小
        return self.config.epsilon / (self.config.federated_rounds * 10)
    
    @property
    def noise_multiplier(self) -> float:
        """获取噪声乘数"""
        return self.privacy_accountant.compute_noise_multiplier(
            self.config.epsilon, self.config.delta,
            self.config.federated_rounds, self.config.clients_per_round,
            self.config.clients_per_round * 100
        )


def create_dp_sgd_config(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    apply_to_global: bool = True,
    apply_to_local: bool = False,
    **kwargs
) -> DPSGDConfig:
    """
    创建DP-SGD配置的便捷函数
    
    Args:
        epsilon: 隐私预算
        delta: 失败概率
        max_grad_norm: 梯度裁剪阈值
        apply_to_global: 是否对全局适配器应用DP-SGD
        apply_to_local: 是否对本地适配器应用DP-SGD
        **kwargs: 其他配置参数
        
    Returns:
        DPSGDConfig实例
    """
    config = DPSGDConfig(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        apply_to_global=apply_to_global,
        apply_to_local=apply_to_local,
        **kwargs
    )
    return config


def create_dual_lora_dp_trainer(
    model: nn.Module,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    **kwargs
) -> DualLoRADPSGDTrainer:
    """
    创建双模块LoRA DP-SGD训练器的便捷函数
    
    Args:
        model: 双模块LoRA模型
        epsilon: 隐私预算
        delta: 失败概率
        max_grad_norm: 梯度裁剪阈值
        **kwargs: 其他配置参数
        
    Returns:
        DualLoRADPSGDTrainer实例
    """
    config = create_dp_sgd_config(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        **kwargs
    )
    return DualLoRADPSGDTrainer(model, config)
