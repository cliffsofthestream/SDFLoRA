import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora import LoraLayer, Linear as LoraLinear
from peft.utils import _get_submodules
import logging

logger = logging.getLogger(__name__)


class DualLoraConfig(LoraConfig):
    """
    双模块LoRA配置类
    扩展PEFT的LoraConfig以支持双模块设置
    """
    
    def __init__(
        self,
        global_r: int = 8,
        local_r: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        global_aggregation_strategy: str = "fedavg",
        local_personalization_strategy: str = "local_only",
        target_modules: Optional[Union[List[str], str]] = None,
        **kwargs
    ):
        # 使用global_r作为基础r参数初始化父类
        super().__init__(
            r=global_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            **kwargs
        )
        
        # 双模块特定配置
        self.global_r = global_r
        self.local_r = local_r
        self.fusion_method = fusion_method
        self.global_aggregation_strategy = global_aggregation_strategy
        self.local_personalization_strategy = local_personalization_strategy
        
        # 设置任务类型（如果未指定）
        if not hasattr(self, 'task_type') or self.task_type is None:
            self.task_type = TaskType.SEQ_CLS


class DualLoraLinear(nn.Module):
    """
    双模块LoRA线性层
    包含全局适配器和本地适配器
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        global_r: int = 8,
        local_r: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        **kwargs
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.adapter_name = adapter_name
        self.global_r = global_r
        self.local_r = local_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.fusion_method = fusion_method
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # 全局适配器（用于联邦聚合）
        self.global_lora_A = nn.Linear(self.in_features, global_r, bias=False)
        self.global_lora_B = nn.Linear(global_r, self.out_features, bias=False)
        self.global_dropout = nn.Dropout(lora_dropout)
        
        # 本地适配器（用于个性化）
        self.local_lora_A = nn.Linear(self.in_features, local_r, bias=False)
        self.local_lora_B = nn.Linear(local_r, self.out_features, bias=False)
        self.local_dropout = nn.Dropout(lora_dropout)
        
        # 融合机制
        if fusion_method == "weighted_sum":
            self.global_weight = nn.Parameter(torch.tensor(0.7))
            self.local_weight = nn.Parameter(torch.tensor(0.3))
        elif fusion_method == "gating":
            self.gate = nn.Sequential(
                nn.Linear(self.out_features * 2, self.out_features),
                nn.Sigmoid()
            )
        
        # 缩放因子
        self.global_scaling = lora_alpha / global_r
        self.local_scaling = lora_alpha / local_r
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化LoRA参数"""
        # 全局适配器初始化
        nn.init.kaiming_uniform_(self.global_lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.global_lora_B.weight)
        
        # 本地适配器初始化
        nn.init.kaiming_uniform_(self.local_lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.local_lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 基础层输出
        base_result = self.base_layer(x)
        
        # 全局适配器
        global_h = self.global_dropout(self.global_lora_A(x))
        global_result = self.global_lora_B(global_h) * self.global_scaling
        
        # 本地适配器
        local_h = self.local_dropout(self.local_lora_A(x))
        local_result = self.local_lora_B(local_h) * self.local_scaling
        
        # 融合两个适配器的输出
        if self.fusion_method == "weighted_sum":
            # 加权求和
            global_w = torch.sigmoid(self.global_weight)
            local_w = torch.sigmoid(self.local_weight)
            # 归一化权重
            total_w = global_w + local_w
            global_w = global_w / total_w
            local_w = local_w / total_w
            
            lora_result = global_w * global_result + local_w * local_result
        elif self.fusion_method == "gating":
            # 门控机制
            concatenated = torch.cat([global_result, local_result], dim=-1)
            gate_weights = self.gate(concatenated)
            lora_result = gate_weights * global_result + (1 - gate_weights) * local_result
        else:
            # 默认简单相加
            lora_result = global_result + local_result
        
        return base_result + lora_result
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """获取全局适配器参数"""
        return {
            f"{self.adapter_name}.global_lora_A.weight": self.global_lora_A.weight,
            f"{self.adapter_name}.global_lora_B.weight": self.global_lora_B.weight,
        }
    
    def get_local_parameters(self) -> Dict[str, torch.Tensor]:
        """获取本地适配器参数"""
        params = {
            f"{self.adapter_name}.local_lora_A.weight": self.local_lora_A.weight,
            f"{self.adapter_name}.local_lora_B.weight": self.local_lora_B.weight,
        }
        
        # 添加融合机制参数
        if self.fusion_method == "weighted_sum":
            params.update({
                f"{self.adapter_name}.global_weight": self.global_weight,
                f"{self.adapter_name}.local_weight": self.local_weight,
            })
        elif self.fusion_method == "gating":
            for name, param in self.gate.named_parameters():
                params[f"{self.adapter_name}.gate.{name}"] = param
        
        return params


class DualLoraPeftModel(nn.Module):
    """
    基于PEFT的双模块LoRA模型
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DualLoraConfig,
        adapter_name: str = "default"
    ):
        super().__init__()
        
        self.model = model
        self.config = config
        self.adapter_name = adapter_name
        self.peft_config = {adapter_name: config}
        
        # 确保num_labels在模型的所有位置都正确设置
        self._ensure_num_labels_consistent()
        
        # 应用双模块LoRA
        self._apply_dual_lora()
        
        # 冻结基础模型参数
        self._freeze_base_model()
    
    def _ensure_num_labels_consistent(self):
        """确保模型的num_labels在所有位置都一致"""
        # 尝试从模型获取num_labels
        num_labels = None
        
        # 方法1: 从config获取
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_labels'):
            num_labels = self.model.config.num_labels
            logger.debug(f"Found num_labels in model.config: {num_labels}")
        
        # 方法2: 从模型属性获取
        if num_labels is None and hasattr(self.model, 'num_labels'):
            num_labels = self.model.num_labels
            logger.debug(f"Found num_labels in model attribute: {num_labels}")
        
        # 方法3: 从分类器层推断
        if num_labels is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if 'classifier' in name.lower() or 'score' in name.lower() or 'head' in name.lower():
                        num_labels = module.out_features
                        logger.debug(f"Inferred num_labels from {name}: {num_labels}")
                        break
        
        # 如果找到了num_labels，确保所有位置都一致
        if num_labels is not None:
            # 更新config
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_labels'):
                if self.model.config.num_labels != num_labels:
                    logger.info(f"Updating model.config.num_labels: {self.model.config.num_labels} -> {num_labels}")
                    self.model.config.num_labels = num_labels
            
            # 更新模型属性
            if hasattr(self.model, 'num_labels'):
                if self.model.num_labels != num_labels:
                    logger.info(f"Updating model.num_labels: {self.model.num_labels} -> {num_labels}")
                    self.model.num_labels = num_labels
            
            # 递归更新所有子模块
            self._recursively_update_num_labels(self.model, num_labels)
    
    def _recursively_update_num_labels(self, module: nn.Module, num_labels: int):
        """递归更新模块及其子模块的num_labels"""
        if hasattr(module, 'config') and hasattr(module.config, 'num_labels'):
            if module.config.num_labels != num_labels:
                logger.debug(f"Updating {type(module).__name__}.config.num_labels: {module.config.num_labels} -> {num_labels}")
                module.config.num_labels = num_labels
        
        if hasattr(module, 'num_labels'):
            if module.num_labels != num_labels:
                logger.debug(f"Updating {type(module).__name__}.num_labels: {module.num_labels} -> {num_labels}")
                module.num_labels = num_labels
        
        # 递归处理子模块
        for child in module.children():
            self._recursively_update_num_labels(child, num_labels)
    
    def _apply_dual_lora(self):
        """应用双模块LoRA到目标模块"""
        target_modules = self.config.target_modules
        if target_modules is None:
            # 默认目标模块
            target_modules = ["query", "value", "key", "dense"]
        elif isinstance(target_modules, str):
            target_modules = [target_modules]
        
        for name, module in self.model.named_modules():
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # 替换为双模块LoRA层
                    dual_lora_layer = DualLoraLinear(
                        base_layer=module,
                        adapter_name=self.adapter_name,
                        global_r=self.config.global_r,
                        local_r=self.config.local_r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        fusion_method=self.config.fusion_method
                    )
                    
                    # 替换模块
                    parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                    if parent_name:
                        parent_module = dict(self.model.named_modules())[parent_name]
                        setattr(parent_module, child_name, dual_lora_layer)
                    else:
                        setattr(self.model, child_name, dual_lora_layer)
    
    def _freeze_base_model(self):
        """冻结基础模型参数"""
        for name, param in self.model.named_parameters():
            if not any(adapter_param in name for adapter_param in 
                      ['global_lora', 'local_lora', 'global_weight', 'local_weight', 'gate']):
                param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    def get_global_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取全局参数状态字典"""
        global_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, DualLoraLinear):
                global_params = module.get_global_parameters()
                for param_name, param_value in global_params.items():
                    global_state[f"{name}.{param_name}"] = param_value
        return global_state
    
    def get_local_state_dict(self) -> Dict[str, torch.Tensor]:
        """获取本地参数状态字典"""
        local_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, DualLoraLinear):
                local_params = module.get_local_parameters()
                for param_name, param_value in local_params.items():
                    local_state[f"{name}.{param_name}"] = param_value
        return local_state
    
    def state_dict(self, return_trainable=True, *args, **kwargs):
        """返回状态字典"""
        if return_trainable:
            # 只返回可训练参数
            trainable_state = {}
            trainable_state.update(self.get_global_state_dict())
            trainable_state.update(self.get_local_state_dict())
            return trainable_state
        else:
            return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False):
        """加载状态字典"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        global_params = sum(p.numel() for p in self.get_global_state_dict().values())
        local_params = sum(p.numel() for p in self.get_local_state_dict().values())
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Dual-LoRA Trainable Parameters:")
        print(f"  Global Adapter: {global_params:,} parameters")
        print(f"  Local Adapter: {local_params:,} parameters")
        print(f"  Total Trainable: {global_params + local_params:,} parameters")
        print(f"  Base Model: {total_params:,} parameters")
        print(f"  Trainable Ratio: {(global_params + local_params) / total_params * 100:.2f}%")
    
    def generate(self, *args, **kwargs):
        """生成方法（如果基础模型支持）"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("Base model does not support generation")


def get_dual_lora_model(
    model: nn.Module,
    config: DualLoraConfig,
    adapter_name: str = "default"
) -> DualLoraPeftModel:
    """
    创建双模块LoRA模型的便捷函数
    
    Args:
        model: 基础预训练模型
        config: 双模块LoRA配置
        adapter_name: 适配器名称
        
    Returns:
        DualLoraPeftModel实例
    """
    return DualLoraPeftModel(model, config, adapter_name)


class DualLoraAdapterModel(nn.Module):
    """
    双模块LoRA适配器模型包装器
    兼容原有的AdapterModel接口
    """
    
    def __init__(
        self,
        model: nn.Module,
        use_adapter: bool = True,
        adapter_package: str = "dual_lora",
        adapter_method: str = "dual_lora",
        global_r: int = 8,
        local_r: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        fusion_method: str = "weighted_sum",
        target_modules: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__()
        
        if use_adapter and adapter_method == "dual_lora":
            # 创建双模块LoRA配置
            config = DualLoraConfig(
                global_r=global_r,
                local_r=local_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fusion_method=fusion_method,
                target_modules=target_modules or ["query", "value", "key", "dense"],
                **kwargs
            )
            
            # 创建双模块LoRA模型
            self.model = get_dual_lora_model(model, config)
        else:
            # 回退到原始模型
            self.model = model
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """生成方法"""
        if hasattr(self.model, 'generate'):
            return self.model.generate(*args, **kwargs)
        else:
            raise NotImplementedError("Model does not support generation")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """状态字典"""
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        else:
            return super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict, strict=False):
        """加载状态字典"""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def get_trainable_state_dict(self):
        """获取可训练参数状态字典"""
        if hasattr(self.model, 'get_global_state_dict') and hasattr(self.model, 'get_local_state_dict'):
            trainable_state = {}
            trainable_state.update(self.model.get_global_state_dict())
            trainable_state.update(self.model.get_local_state_dict())
            return trainable_state
        else:
            # 回退到原始方法
            grad_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    grad_params.append(name)
            
            model_state_dict = self.model.state_dict()
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k in grad_params:
                    new_state_dict[k] = v
            return new_state_dict
    
    def save_model(self, path, state=0):
        """保存模型"""
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)
    
    def print_trainable_parameters(self):
        """打印可训练参数"""
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    def __getattr__(self, name):
        """属性转发到内部模型"""
        # 避免无限递归：直接检查内部模型的__dict__
        if name in self.__dict__:
            return self.__dict__[name]
        
        # 如果访问的是model属性，直接返回内部模型
        if name == 'model':
            # 从_modules中获取model
            if hasattr(self, '_modules') and 'model' in self._modules:
                return self._modules['model']
            else:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # 直接访问内部模型的属性，避免使用hasattr
        try:
            # 使用getattr而不是object.__getattribute__来避免递归
            return getattr(self.__dict__['model'], name)
        except (AttributeError, KeyError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# 兼容函数
def enable_dual_lora_adapter(
    model: nn.Module,
    global_r: int = 8,
    local_r: int = 4,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    fusion_method: str = "weighted_sum",
    target_modules: Optional[List[str]] = None,
    **kwargs
) -> DualLoraPeftModel:
    """
    为模型启用双模块LoRA适配器的便捷函数
    
    Args:
        model: 基础模型
        global_r: 全局适配器rank
        local_r: 本地适配器rank
        lora_alpha: LoRA alpha参数
        lora_dropout: dropout率
        fusion_method: 融合方法
        target_modules: 目标模块列表
        **kwargs: 其他参数
        
    Returns:
        DualLoraPeftModel实例
    """
    config = DualLoraConfig(
        global_r=global_r,
        local_r=local_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fusion_method=fusion_method,
        target_modules=target_modules or ["query", "value", "key", "dense"],
        **kwargs
    )
    
    return get_dual_lora_model(model, config)
