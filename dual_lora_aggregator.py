"""
双模块LoRA聚合器实现（带DP-SGD支持）
支持全局适配器的联邦聚合和本地适配器的个性化保持

参考: 
- NeurIPS 2024 - Dual-Personalizing Adapter for Federated Foundation Models
- "IMPROVING LORA IN PRIVACY-PRESERVING FEDERATED LEARNING"论文
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class DualLoRAAggregator:
    """
    双模块LoRA聚合器
    
    核心思想：
    1. 全局适配器参数参与联邦聚合，学习跨客户端的共享知识
    2. 本地适配器参数保持本地，处理客户端特定的个性化需求
    3. 支持多种聚合策略和个性化机制
    """
    
    def __init__(
        self,
        model=None,
        config=None,
        device='cpu',
        global_aggregation_strategy: str = "fedavg",  # "fedavg", "stacked", "attention_weighted"
        local_personalization_strategy: str = "local_only",  # "local_only", "meta_learning", "adaptive"
        client_ranks: Optional[Dict[int, Tuple[int, int]]] = None,  # {client_id: (global_rank, local_rank)}
        enable_stacking: bool = False,
        enable_zero_padding: bool = False,
        enable_heterogeneous: bool = True,
        enable_dp_sgd: bool = False,  # 是否启用DP-SGD
        dp_config: Optional[Dict] = None,  # DP-SGD配置
        client_scaling_factor: float = 0.1,  # 客户端权重缩放因子(pk)
        use_fixed_scaling: bool = False  # 是否使用固定缩放因子而非基于数据量的权重
    ):
        self.model = model
        self.config = config
        self.device = device
        self.global_aggregation_strategy = global_aggregation_strategy
        self.local_personalization_strategy = local_personalization_strategy
        self.enable_stacking = enable_stacking
        self.enable_zero_padding = enable_zero_padding
        self.enable_heterogeneous = enable_heterogeneous
        self.enable_dp_sgd = enable_dp_sgd
        self.dp_config = dp_config or {}
        self.client_scaling_factor = client_scaling_factor
        self.use_fixed_scaling = use_fixed_scaling
        
        # 客户端rank配置 {client_id: (global_rank, local_rank)}
        self.client_ranks = client_ranks or {}
        
        # 计算全局总rank（用于堆叠）
        if self.client_ranks:
            self.global_total_rank = sum(ranks[0] for ranks in self.client_ranks.values())
            self.local_total_rank = sum(ranks[1] for ranks in self.client_ranks.values())
        else:
            self.global_total_rank = 0
            self.local_total_rank = 0
        
        logger.info(f"DualLoRA Aggregator initialized:")
        logger.info(f"  Global aggregation strategy: {global_aggregation_strategy}")
        logger.info(f"  Local personalization strategy: {local_personalization_strategy}")
        logger.info(f"  Client ranks: {self.client_ranks}")
        logger.info(f"  Global total rank: {self.global_total_rank}")
        logger.info(f"  Local total rank: {self.local_total_rank}")
        logger.info(f"  Client scaling factor (pk): {self.client_scaling_factor}")
        logger.info(f"  Use fixed scaling: {self.use_fixed_scaling}")
    
    def aggregate(self, agg_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        双模块LoRA聚合主函数
        
        Args:
            agg_info: 聚合信息，包含客户端反馈
                格式: {"client_feedback": [(client_id, (sample_size, model_state_dict)), ...]}
        
        Returns:
            聚合后的全局模型状态字典
        """
        feedback = agg_info["client_feedback"]
        client_ids, sizes, models = self._parse_feedback(feedback)
        
        # 计算权重
        if self.use_fixed_scaling:
            # 使用固定缩放因子(pk)作为权重
            weights = [self.client_scaling_factor] * len(sizes)
            logger.info(f"Using fixed client scaling factor: {self.client_scaling_factor}")
        else:
            # 使用基于数据量的权重（原有逻辑）
            total_size = sum(sizes)
            weights = [s / total_size for s in sizes]
            logger.info(f"Using data-size based weights: {weights}")
        
        # 分离全局和本地参数
        global_params, local_params, other_params = self._separate_parameters(models)
        
        # 聚合全局适配器参数（带DP-SGD支持）
        aggregated_global = self._aggregate_global_parameters_with_privacy(
            global_params, client_ids, weights
        )
        
        # 处理本地适配器参数（通常不聚合，但可能需要一些个性化策略）
        processed_local = self._process_local_parameters(
            local_params, client_ids, weights
        )
        
        # 聚合其他参数（如分类器头等）
        aggregated_other = self._aggregate_other_parameters(
            other_params, weights
        )
        
        # 合并所有参数
        global_dict = {}
        global_dict.update(aggregated_global)
        global_dict.update(processed_local)
        global_dict.update(aggregated_other)
        
        # 更新模型
        if self.model is not None:
            self._update_model(global_dict)
        
        logger.info(f"Aggregated {len(aggregated_global)} global parameters, "
                   f"{len(processed_local)} local parameters, "
                   f"{len(aggregated_other)} other parameters")
        
        return global_dict
    
    def _parse_feedback(self, feedback: List[Tuple]) -> Tuple[List[int], List[float], List[Dict]]:
        """解析客户端反馈"""
        client_ids, sizes, models = [], [], []
        
        for entry in feedback:
            if len(entry) == 2:
                cid, content = entry
                if isinstance(content, tuple) and len(content) == 2:
                    sz, state = content
                else:
                    sz, state = 1, content
            else:
                raise ValueError(f"Unexpected feedback entry format: {entry}")
            
            client_ids.append(cid)
            sizes.append(sz)
            models.append(state)
        
        return client_ids, sizes, models
    
    def _separate_parameters(
        self, 
        models: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        分离全局、本地和其他参数
        
        Returns:
            (global_params_list, local_params_list, other_params_list)
        """
        global_params_list = []
        local_params_list = []
        other_params_list = []
        
        for model_state in models:
            global_params = {}
            local_params = {}
            other_params = {}
            
            for key, value in model_state.items():
                if self._is_global_parameter(key):
                    global_params[key] = value.cpu()
                elif self._is_local_parameter(key):
                    local_params[key] = value.cpu()
                else:
                    other_params[key] = value.cpu()
            
            global_params_list.append(global_params)
            local_params_list.append(local_params)
            other_params_list.append(other_params)
        
        return global_params_list, local_params_list, other_params_list
    
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
    
    def _aggregate_global_parameters(
        self,
        global_params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """聚合全局适配器参数"""
        if not global_params_list or not global_params_list[0]:
            return {}
        
        if self.global_aggregation_strategy == "fedavg":
            return self._fedavg_aggregation(global_params_list, weights)
        elif self.global_aggregation_strategy == "stacked":
            return self._stacked_aggregation(global_params_list, client_ids, weights)
        elif self.global_aggregation_strategy == "attention_weighted":
            return self._attention_weighted_aggregation(global_params_list, weights)
        else:
            logger.warning(f"Unknown global aggregation strategy: {self.global_aggregation_strategy}, "
                          f"falling back to FedAvg")
            return self._fedavg_aggregation(global_params_list, weights)
    
    def _fedavg_aggregation(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """标准FedAvg聚合"""
        if not params_list:
            return {}
        
        aggregated = {}
        template = params_list[0]
        
        for key in template.keys():
            aggregated[key] = torch.zeros_like(template[key])
            for params, weight in zip(params_list, weights):
                if key in params:
                    aggregated[key] += params[key] * weight
        
        return aggregated
    
    def _stacked_aggregation(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """堆叠聚合（类似FedSA-LoRA的堆叠机制）"""
        if not params_list or not self.enable_stacking:
            return self._fedavg_aggregation(params_list, weights)
        
        aggregated = {}
        template = params_list[0]
        
        for key in template.keys():
            if "global_lora_A" in key:
                # A矩阵按行堆叠
                matrices = []
                for cid, params in zip(client_ids, params_list):
                    if key in params:
                        matrix = params[key]
                        if self.enable_heterogeneous and cid in self.client_ranks:
                            # 异构情况下可能需要填充
                            global_rank = self.client_ranks[cid][0]
                            if self.enable_zero_padding:
                                max_rank = max(ranks[0] for ranks in self.client_ranks.values())
                                pad_rows = max_rank - global_rank
                                matrix = F.pad(matrix, (0, 0, 0, pad_rows))
                        matrices.append(matrix)
                
                aggregated[key] = torch.cat(matrices, dim=0)
                
            elif "global_lora_B" in key:
                # B矩阵按列堆叠
                matrices = []
                for cid, params in zip(client_ids, params_list):
                    if key in params:
                        matrix = params[key]
                        if self.enable_heterogeneous and cid in self.client_ranks:
                            # 异构情况下可能需要填充
                            global_rank = self.client_ranks[cid][0]
                            if self.enable_zero_padding:
                                max_rank = max(ranks[0] for ranks in self.client_ranks.values())
                                pad_cols = max_rank - global_rank
                                matrix = F.pad(matrix, (0, pad_cols, 0, 0))
                        matrices.append(matrix)
                
                aggregated[key] = torch.cat(matrices, dim=1)
            else:
                # 其他参数使用FedAvg
                aggregated[key] = torch.zeros_like(template[key])
                for params, weight in zip(params_list, weights):
                    if key in params:
                        aggregated[key] += params[key] * weight
        
        return aggregated
    
    def _aggregate_global_parameters_with_privacy(
        self,
        global_params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """带隐私保护的全局适配器参数聚合"""
        if not global_params_list or not global_params_list[0]:
            return {}
        
        # 执行标准聚合
        if self.global_aggregation_strategy == "fedavg":
            aggregated = self._fedavg_aggregation(global_params_list, weights)
        elif self.global_aggregation_strategy == "stacked":
            aggregated = self._stacked_aggregation(global_params_list, client_ids, weights)
        elif self.global_aggregation_strategy == "attention_weighted":
            aggregated = self._attention_weighted_aggregation(global_params_list, weights)
        else:
            logger.warning(f"Unknown global aggregation strategy: {self.global_aggregation_strategy}, "
                          f"falling back to FedAvg")
            aggregated = self._fedavg_aggregation(global_params_list, weights)
        
        # 如果启用DP-SGD，添加聚合噪声
        if self.enable_dp_sgd and self.dp_config.get('enable_secure_aggregation', True):
            aggregated = self._add_aggregation_noise(aggregated, len(client_ids))
        
        return aggregated
    
    def _add_aggregation_noise(
        self, 
        aggregated_params: Dict[str, torch.Tensor], 
        num_clients: int
    ) -> Dict[str, torch.Tensor]:
        """添加聚合噪声"""
        if not self.dp_config:
            return aggregated_params
        
        # 获取噪声参数
        noise_scale = self.dp_config.get('aggregation_noise_scale', 0.8)
        max_grad_norm = self.dp_config.get('max_grad_norm', 1.0)
        epsilon = self.dp_config.get('epsilon', 1.0)
        
        # 计算聚合噪声标准差
        # 使用高斯机制：噪声标准差 = (2 * ln(1.25/δ) * Δf) / ε
        # 其中Δf是敏感度，这里使用max_grad_norm作为近似
        noise_std = (2 * np.log(1.25 / self.dp_config.get('delta', 1e-5)) * 
                    max_grad_norm * noise_scale) / epsilon
        
        # 添加噪声到每个参数
        noisy_params = {}
        for key, param in aggregated_params.items():
            if self._is_global_parameter(key):
                noise = torch.normal(0, noise_std, param.shape, device=param.device)
                noisy_params[key] = param + noise
            else:
                noisy_params[key] = param
        
        logger.info(f"Added aggregation noise with std={noise_std:.6f} to {len(noisy_params)} parameters")
        return noisy_params
    
    def _attention_weighted_aggregation(
        self,
        params_list: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """基于注意力机制的加权聚合"""
        if not params_list:
            return {}
        
        # 计算参数相似性作为注意力权重
        aggregated = {}
        template = params_list[0]
        
        for key in template.keys():
            matrices = [params[key] for params in params_list if key in params]
            if not matrices:
                continue
            
            # 计算注意力权重
            attention_weights = self._compute_attention_weights(matrices, weights)
            
            # 加权聚合
            aggregated[key] = torch.zeros_like(matrices[0])
            for matrix, att_weight in zip(matrices, attention_weights):
                aggregated[key] += matrix * att_weight
        
        return aggregated
    
    def _compute_attention_weights(
        self,
        matrices: List[torch.Tensor],
        base_weights: List[float]
    ) -> List[float]:
        """计算注意力权重"""
        if len(matrices) <= 1:
            return base_weights
        
        # 计算矩阵间的余弦相似性
        similarities = []
        for i, mat_i in enumerate(matrices):
            sim_i = 0.0
            for j, mat_j in enumerate(matrices):
                if i != j:
                    # 展平矩阵并计算余弦相似性
                    flat_i = mat_i.flatten()
                    flat_j = mat_j.flatten()
                    cos_sim = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0))
                    sim_i += cos_sim.item()
            similarities.append(sim_i / (len(matrices) - 1))
        
        # 结合基础权重和相似性计算最终权重
        attention_weights = []
        for base_w, sim in zip(base_weights, similarities):
            att_w = base_w * (1 + sim)  # 相似性高的客户端获得更高权重
            attention_weights.append(att_w)
        
        # 归一化
        total_weight = sum(attention_weights)
        attention_weights = [w / total_weight for w in attention_weights]
        
        return attention_weights
    
    def _process_local_parameters(
        self,
        local_params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """处理本地适配器参数"""
        if not local_params_list:
            return {}
        
        if self.local_personalization_strategy == "local_only":
            # 本地参数不参与聚合，返回空字典
            return {}
        elif self.local_personalization_strategy == "meta_learning":
            # 使用元学习策略处理本地参数
            return self._meta_learning_local_processing(local_params_list, weights)
        elif self.local_personalization_strategy == "adaptive":
            # 自适应策略
            return self._adaptive_local_processing(local_params_list, client_ids, weights)
        else:
            logger.warning(f"Unknown local personalization strategy: {self.local_personalization_strategy}")
            return {}
    
    def _meta_learning_local_processing(
        self,
        local_params_list: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """元学习策略处理本地参数"""
        # 计算本地参数的元梯度或元更新
        # 这里实现一个简化版本：计算加权平均作为元初始化
        if not local_params_list or not local_params_list[0]:
            return {}
        
        meta_params = {}
        template = local_params_list[0]
        
        for key in template.keys():
            meta_params[f"meta_{key}"] = torch.zeros_like(template[key])
            for params, weight in zip(local_params_list, weights):
                if key in params:
                    meta_params[f"meta_{key}"] += params[key] * weight
        
        return meta_params
    
    def _adaptive_local_processing(
        self,
        local_params_list: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """自适应策略处理本地参数"""
        # 根据客户端性能自适应调整本地参数
        # 这里实现一个简化版本
        return {}
    
    def _aggregate_other_parameters(
        self,
        other_params_list: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """聚合其他参数（如分类器头）"""
        return self._fedavg_aggregation(other_params_list, weights)
    
    def _update_model(self, global_dict: Dict[str, torch.Tensor]):
        """更新模型参数"""
        if self.model is None:
            return
        
        try:
            # 尝试使用DualLoRAModel的专用方法
            if hasattr(self.model, 'load_global_state_dict'):
                global_params = {k: v for k, v in global_dict.items() 
                               if self._is_global_parameter(k)}
                self.model.load_global_state_dict(global_params)
            else:
                # 回退到标准方法
                self.model.load_state_dict(global_dict, strict=False)
        except Exception as e:
            logger.warning(f"Failed to update model: {e}")
    
    def update(self, model_parameters: Dict[str, torch.Tensor]):
        """更新模型参数（兼容原有接口）"""
        if self.model is not None:
            self._update_model(model_parameters)
    
    def save_model(self, path: str, cur_round: int = -1):
        """保存模型"""
        if self.model is None:
            return
        
        ckpt = {
            'cur_round': cur_round,
            'model': self.model.state_dict(),
            'aggregator_config': {
                'global_aggregation_strategy': self.global_aggregation_strategy,
                'local_personalization_strategy': self.local_personalization_strategy,
                'client_ranks': self.client_ranks,
            }
        }
        torch.save(ckpt, path)
        logger.info(f"Saved dual-LoRA model to {path}")
    
    def load_model(self, path: str) -> int:
        """加载模型"""
        if self.model is None:
            raise ValueError("Model is None, cannot load state dict")
        
        import os
        if not os.path.exists(path):
            raise ValueError(f"The file {path} does NOT exist")
        
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'], strict=False)
        
        # 加载聚合器配置
        if 'aggregator_config' in ckpt:
            config = ckpt['aggregator_config']
            self.global_aggregation_strategy = config.get(
                'global_aggregation_strategy', self.global_aggregation_strategy
            )
            self.local_personalization_strategy = config.get(
                'local_personalization_strategy', self.local_personalization_strategy
            )
            self.client_ranks = config.get('client_ranks', self.client_ranks)
        
        logger.info(f"Loaded dual-LoRA model from {path}")
        return ckpt.get('cur_round', -1)


class DualLoRAFederatedAggregator(DualLoRAAggregator):
    """
    联邦学习专用的双模块LoRA聚合器
    继承自DualLoRAAggregator，添加联邦学习特定功能
    """
    
    def __init__(
        self,
        model=None,
        config=None,
        device='cpu',
        **kwargs
    ):
        # 从config中提取配置
        if config is not None:
            global_strategy = getattr(config.aggregator, 'global_aggregation_strategy', 'fedavg')
            local_strategy = getattr(config.aggregator, 'local_personalization_strategy', 'local_only')
            
            # 解析客户端ranks配置
            client_ranks = {}
            if hasattr(config.aggregator, 'dual_lora_ranks'):
                raw_ranks = config.aggregator.dual_lora_ranks
                if hasattr(raw_ranks, 'items'):
                    for k, v in raw_ranks.items():
                        if isinstance(k, str) and not k.startswith('_'):
                            try:
                                client_id = int(k)
                                if isinstance(v, (list, tuple)) and len(v) == 2:
                                    client_ranks[client_id] = (int(v[0]), int(v[1]))
                            except (ValueError, TypeError):
                                continue
            
            # 解析DP-SGD配置
            dp_config = {}
            if hasattr(config, 'differential_privacy'):
                dp_config = {
                    'enabled': getattr(config.differential_privacy, 'enabled', False),
                    'epsilon': getattr(config.differential_privacy, 'epsilon', 1.0),
                    'delta': getattr(config.differential_privacy, 'delta', 1e-5),
                    'max_grad_norm': getattr(config.differential_privacy, 'max_grad_norm', 1.0),
                    'noise_multiplier': getattr(config.differential_privacy, 'noise_multiplier', 1.1),
                    'apply_to_global': getattr(config.differential_privacy, 'apply_to_global', True),
                    'apply_to_local': getattr(config.differential_privacy, 'apply_to_local', False),
                    'global_noise_scale': getattr(config.differential_privacy, 'global_noise_scale', 1.0),
                    'local_noise_scale': getattr(config.differential_privacy, 'local_noise_scale', 0.5),
                    'enable_secure_aggregation': getattr(config.differential_privacy, 'enable_secure_aggregation', True),
                    'aggregation_noise_scale': getattr(config.differential_privacy, 'aggregation_noise_scale', 0.8),
                }
            
            # 解析客户端权重缩放因子配置
            client_scaling_factor = getattr(config.aggregator, 'client_scaling_factor', 0.1)
            use_fixed_scaling = getattr(config.aggregator, 'use_fixed_scaling', False)
            
            kwargs.update({
                'global_aggregation_strategy': global_strategy,
                'local_personalization_strategy': local_strategy,
                'client_ranks': client_ranks,
                'enable_stacking': getattr(config.aggregator, 'stacking', False),
                'enable_zero_padding': getattr(config.aggregator, 'zero_padding', False),
                'enable_heterogeneous': getattr(config.aggregator, 'heter', True),
                'enable_dp_sgd': dp_config.get('enabled', False),
                'dp_config': dp_config,
                'client_scaling_factor': client_scaling_factor,
                'use_fixed_scaling': use_fixed_scaling,
            })
        
        super().__init__(model=model, config=config, device=device, **kwargs)
        
        logger.info(f"DualLoRA Federated Aggregator initialized with config-based settings")
