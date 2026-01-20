"""


1. 全局适配器 (Global Adapter): 处理跨客户端共享的通用知识
2. 本地适配器 (Local Adapter): 处理客户端特定的个性化需求
3. 融合机制: 将两个适配器的输出进行有效融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict


class DualLoRALayer(nn.Module):
    """
    双模块LoRA层实现
    包含全局适配器和本地适配器
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        global_rank: int = 8,
        local_rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        fusion_method: str = "weighted_sum"  # "weighted_sum", "attention", "gating"
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.alpha = alpha
        self.dropout = dropout
        self.fusion_method = fusion_method
        
        # 全局适配器 - 用于联邦学习中的全局知识共享
        self.global_lora_A = nn.Linear(in_features, global_rank, bias=False)
        self.global_lora_B = nn.Linear(global_rank, out_features, bias=False)
        self.global_dropout = nn.Dropout(dropout)
        
        # 本地适配器 - 用于客户端特定的个性化
        self.local_lora_A = nn.Linear(in_features, local_rank, bias=False)
        self.local_lora_B = nn.Linear(local_rank, out_features, bias=False)
        self.local_dropout = nn.Dropout(dropout)
        
        # 融合机制
        if fusion_method == "weighted_sum":
            # 可学习的权重参数
            self.global_weight = nn.Parameter(torch.tensor(0.7))
            self.local_weight = nn.Parameter(torch.tensor(0.3))
        elif fusion_method == "attention":
            # 注意力机制融合
            self.attention = nn.MultiheadAttention(
                embed_dim=out_features,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        elif fusion_method == "gating":
            # 门控机制融合
            self.gate = nn.Sequential(
                nn.Linear(out_features * 2, out_features),
                nn.Sigmoid()
            )
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化LoRA参数"""
        # 全局适配器初始化
        nn.init.kaiming_uniform_(self.global_lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.global_lora_B.weight)
        
        # 本地适配器初始化
        nn.init.kaiming_uniform_(self.local_lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.local_lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, in_features]
            
        Returns:
            融合后的输出张量 [batch_size, seq_len, out_features]
        """
        # 全局适配器前向传播
        global_h = self.global_dropout(self.global_lora_A(x))
        global_output = self.global_lora_B(global_h) * (self.alpha / self.global_rank)
        
        # 本地适配器前向传播
        local_h = self.local_dropout(self.local_lora_A(x))
        local_output = self.local_lora_B(local_h) * (self.alpha / self.local_rank)
        
        # 融合两个适配器的输出
        if self.fusion_method == "weighted_sum":
            # 加权求和融合
            fused_output = (
                torch.sigmoid(self.global_weight) * global_output +
                torch.sigmoid(self.local_weight) * local_output
            )
        elif self.fusion_method == "attention":
            # 注意力机制融合
            # 将两个输出作为query, key, value
            combined = torch.stack([global_output, local_output], dim=1)  # [B, 2, L, D]
            batch_size, seq_len = x.shape[0], x.shape[1]
            combined = combined.view(batch_size, 2 * seq_len, -1)
            
            fused_output, _ = self.attention(combined, combined, combined)
            fused_output = fused_output.view(batch_size, 2, seq_len, -1).mean(dim=1)
        elif self.fusion_method == "gating":
            # 门控机制融合
            concatenated = torch.cat([global_output, local_output], dim=-1)
            gate_weights = self.gate(concatenated)
            fused_output = gate_weights * global_output + (1 - gate_weights) * local_output
        else:
            # 默认简单相加
            fused_output = global_output + local_output
        
        return fused_output
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局适配器参数（用于联邦聚合）"""
        return {
            'global_lora_A.weight': self.global_lora_A.weight,
            'global_lora_B.weight': self.global_lora_B.weight,
        }
    
    def get_local_parameters(self) -> Dict[str, torch.Tensor]:
        """获取本地适配器参数（保持本地）"""
        params = {
            'local_lora_A.weight': self.local_lora_A.weight,
            'local_lora_B.weight': self.local_lora_B.weight,
        }
        
        # 添加融合机制的参数
        if self.fusion_method == "weighted_sum":
            params.update({
                'global_weight': self.global_weight,
                'local_weight': self.local_weight,
            })
        elif self.fusion_method == "attention":
            params.update(self.attention.state_dict())
        elif self.fusion_method == "gating":
            params.update(self.gate.state_dict())
        
        return params


class DualLoRAConfig:
    """双模块LoRA配置类"""
    
    def __init__(
        self,
        global_rank: int = 8,
        local_rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        target_modules: Optional[list] = None,
        global_aggregation_strategy: str = "fedavg",  # "fedavg", "stacked"
        local_personalization_strategy: str = "local_only"  # "local_only", "meta_learning"
    ):
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.alpha = alpha
        self.dropout = dropout
        self.fusion_method = fusion_method
        self.target_modules = target_modules or ["query", "value", "key", "dense"]
        self.global_aggregation_strategy = global_aggregation_strategy
        self.local_personalization_strategy = local_personalization_strategy


class DualLoRAModel(nn.Module):
    """
    双模块LoRA模型包装器
    将双模块LoRA应用到预训练模型上
    """
    
    def __init__(self, base_model: nn.Module, config: DualLoRAConfig):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.dual_lora_layers = nn.ModuleDict()
        
        # 应用双模块LoRA到目标模块
        self._apply_dual_lora()
        
        # 冻结基础模型参数
        self._freeze_base_model()
    
    def _apply_dual_lora(self):
        """将双模块LoRA应用到目标模块"""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    # 创建双模块LoRA层
                    dual_lora = DualLoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        global_rank=self.config.global_rank,
                        local_rank=self.config.local_rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                        fusion_method=self.config.fusion_method
                    )
                    
                    self.dual_lora_layers[name] = dual_lora
    
    def _freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        # 这里需要根据具体的模型结构来实现
        # 一般需要hook机制来插入LoRA层
        return self._forward_with_dual_lora(*args, **kwargs)
    
    def _forward_with_dual_lora(self, *args, **kwargs):
        """带双模块LoRA的前向传播"""
        # 注册前向hook来插入LoRA计算
        hooks = []
        
        def create_hook(lora_layer, original_module):
            def hook_fn(module, input, output):
                # 获取原始输出
                original_output = output
                # 计算LoRA输出
                lora_output = lora_layer(input[0])
                # 返回融合结果
                return original_output + lora_output
            return hook_fn
        
        # 为每个目标模块注册hook
        for name, lora_layer in self.dual_lora_layers.items():
            target_module = dict(self.base_model.named_modules())[name]
            hook = target_module.register_forward_hook(
                create_hook(lora_layer, target_module)
            )
            hooks.append(hook)
        
        try:
            # 执行前向传播
            output = self.base_model(*args, **kwargs)
        finally:
            # 清理hooks
            for hook in hooks:
                hook.remove()
        
        return output
    
    def get_global_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取全局参数状态字典（用于联邦聚合）"""
        global_state = {}
        for name, lora_layer in self.dual_lora_layers.items():
            layer_params = lora_layer.get_global_parameters()
            for param_name, param_value in layer_params.items():
                global_state[f"{name}.{param_name}"] = param_value
        return global_state
    
    def get_local_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取本地参数状态字典（保持本地）"""
        local_state = {}
        for name, lora_layer in self.dual_lora_layers.items():
            layer_params = lora_layer.get_local_parameters()
            for param_name, param_value in layer_params.items():
                local_state[f"{name}.{param_name}"] = param_value
        return local_state
    
    def load_global_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """加载全局参数状态字典"""
        for name, lora_layer in self.dual_lora_layers.items():
            layer_prefix = f"{name}."
            layer_state = {
                k[len(layer_prefix):]: v 
                for k, v in state_dict.items() 
                if k.startswith(layer_prefix)
            }
            
            # 只加载全局参数
            if 'global_lora_A.weight' in layer_state:
                lora_layer.global_lora_A.weight.data = layer_state['global_lora_A.weight']
            if 'global_lora_B.weight' in layer_state:
                lora_layer.global_lora_B.weight.data = layer_state['global_lora_B.weight']
    
    def load_local_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """加载本地参数状态字典"""
        for name, lora_layer in self.dual_lora_layers.items():
            layer_prefix = f"{name}."
            layer_state = {
                k[len(layer_prefix):]: v 
                for k, v in state_dict.items() 
                if k.startswith(layer_prefix)
            }
            
            # 加载本地参数
            if 'local_lora_A.weight' in layer_state:
                lora_layer.local_lora_A.weight.data = layer_state['local_lora_A.weight']
            if 'local_lora_B.weight' in layer_state:
                lora_layer.local_lora_B.weight.data = layer_state['local_lora_B.weight']
            
            # 加载融合机制参数
            if lora_layer.fusion_method == "weighted_sum":
                if 'global_weight' in layer_state:
                    lora_layer.global_weight.data = layer_state['global_weight']
                if 'local_weight' in layer_state:
                    lora_layer.local_weight.data = layer_state['local_weight']
    
    def state_dict(self, return_trainable=True, *args, **kwargs):
        """返回模型状态字典"""
        if return_trainable:
            # 返回所有可训练参数（全局+本地）
            trainable_state = {}
            trainable_state.update(self.get_global_state_dict())
            trainable_state.update(self.get_local_state_dict())
            return trainable_state
        else:
            # 返回完整模型状态
            full_state = self.base_model.state_dict(*args, **kwargs)
            full_state.update(self.get_global_state_dict())
            full_state.update(self.get_local_state_dict())
            return full_state
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        global_params = sum(p.numel() for p in self.get_global_state_dict().values())
        local_params = sum(p.numel() for p in self.get_local_state_dict().values())
        total_params = sum(p.numel() for p in self.base_model.parameters())
        
        print(f"Dual-LoRA Trainable Parameters:")
        print(f"  Global Adapter: {global_params:,} parameters")
        print(f"  Local Adapter: {local_params:,} parameters")
        print(f"  Total Trainable: {global_params + local_params:,} parameters")
        print(f"  Base Model: {total_params:,} parameters")
        print(f"  Trainable Ratio: {(global_params + local_params) / total_params * 100:.2f}%")


def create_dual_lora_model(
    base_model: nn.Module,
    global_rank: int = 8,
    local_rank: int = 4,
    alpha: float = 16.0,
    dropout: float = 0.1,
    fusion_method: str = "weighted_sum",
    target_modules: Optional[list] = None
) -> DualLoRAModel:
    """
    创建双模块LoRA模型的便捷函数
    
    Args:
        base_model: 基础预训练模型
        global_rank: 全局适配器的rank
        local_rank: 本地适配器的rank
        alpha: LoRA缩放参数
        dropout: dropout率
        fusion_method: 融合方法
        target_modules: 目标模块列表
        
    Returns:
        DualLoRAModel实例
    """
    config = DualLoRAConfig(
        global_rank=global_rank,
        local_rank=local_rank,
        alpha=alpha,
        dropout=dropout,
        fusion_method=fusion_method,
        target_modules=target_modules
    )
    
    return DualLoRAModel(base_model, config)
