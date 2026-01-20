import torch
import os
from transformers import AutoModelForSequenceClassification, AutoConfig
from dual_lora_peft_adapter import DualLoraAdapterModel, DualLoraConfig, enable_dual_lora_adapter
import logging

logger = logging.getLogger(__name__)


def get_model_from_huggingface_dual_lora(model_name, config):
    """
    从HuggingFace加载模型并应用双模块LoRA
    
    Args:
        model_name: 模型名称
        config: 配置对象
        
    Returns:
        应用了双模块LoRA的模型
    """
    kwargs = {}
    # 检查缓存配置是否存在
    if (hasattr(config, 'llm') and 
        hasattr(config.llm, 'cache') and 
        hasattr(config.llm.cache, 'model') and 
        len(config.llm.cache.model)):
        kwargs['cache_dir'] = config.llm.cache.model
    
    # 添加GLUE任务的标签数量
    # 优先级：label_list > dataset type > config.data.num_labels
    # label_list最可靠，因为它直接反映了实际的标签数量
    expected_num_labels = None
    
    # 详细的调试信息
    if hasattr(config, 'data'):
        logger.info(f"DEBUG: config.data exists, checking attributes...")
        
        # 最高优先级：使用label_list（最可靠，即使type被错误修改）
        # 尝试多种方式访问label_list
        label_list_value = None
        try:
            # 方式1：直接属性访问
            if hasattr(config.data, 'label_list'):
                label_list_value = getattr(config.data, 'label_list', None)
                logger.info(f"DEBUG: Found label_list via hasattr: {label_list_value} (type: {type(label_list_value)})")
        except Exception as e:
            logger.warning(f"DEBUG: Error accessing label_list via hasattr: {e}")
        
        # 如果方式1失败，尝试方式2：字典访问（某些配置对象可能是dict-like）
        if not label_list_value:
            try:
                if isinstance(config.data, dict) or hasattr(config.data, '__getitem__'):
                    label_list_value = config.data.get('label_list') if hasattr(config.data, 'get') else config.data['label_list']
                    logger.info(f"DEBUG: Found label_list via dict access: {label_list_value}")
            except Exception as e:
                logger.warning(f"DEBUG: Error accessing label_list via dict: {e}")
        
        # 如果方式2失败，尝试方式3：通过__dict__访问
        if not label_list_value:
            try:
                if hasattr(config.data, '__dict__'):
                    label_list_value = config.data.__dict__.get('label_list')
                    logger.info(f"DEBUG: Found label_list via __dict__: {label_list_value}")
            except Exception as e:
                logger.warning(f"DEBUG: Error accessing label_list via __dict__: {e}")
        
        # 处理找到的label_list
        if label_list_value:
            try:
                label_list_num = len(label_list_value)
                logger.info(f"DEBUG: label_list length = {label_list_num}")
                if label_list_num > 0:
                    expected_num_labels = label_list_num
                    logger.info(f"✓ Inferred num_labels from label_list ({label_list_value}): {expected_num_labels} (HIGHEST PRIORITY)")
                else:
                    logger.warning(f"DEBUG: label_list exists but is empty!")
            except Exception as e:
                logger.error(f"DEBUG: Error getting label_list length: {e}")
        else:
            logger.warning(f"DEBUG: Could not find label_list using any method")
        
        # 次优先级：从数据集类型推断
        if expected_num_labels is None and hasattr(config.data, 'type'):
            data_type_raw = config.data.type
            data_type = str(data_type_raw).lower()
            logger.info(f"DEBUG: config.data.type = '{data_type_raw}' (lowercase: '{data_type}')")
            
            # 检查MNLI（包括matched和mismatched变体）
            if 'mnli' in data_type:
                expected_num_labels = 3
                logger.info(f"✓ Dataset type is MNLI, num_labels MUST be 3 (forcing override)")
            # 检查二分类任务
            elif 'qnli' in data_type or 'mrpc' in data_type or 'qqp' in data_type or 'sst' in data_type:
                expected_num_labels = 2
                logger.info(f"✓ Dataset type is {data_type}, num_labels MUST be 2 (forcing override)")
            else:
                logger.warning(f"DEBUG: data_type '{data_type}' not recognized in dataset type mapping")
        elif expected_num_labels is None:
            logger.warning(f"DEBUG: config.data.type does not exist")
    else:
        logger.warning(f"DEBUG: config.data does not exist")
    
    # 最低优先级：从配置读取（如果前面都没推断出来）
    # 但在使用config.data.num_labels之前，先检查它是否合理
    if expected_num_labels is None:
        if hasattr(config, 'data') and hasattr(config.data, 'num_labels'):
            config_num_labels = config.data.num_labels
            logger.warning(f"⚠ Could not infer from label_list or dataset type, config.data.num_labels = {config_num_labels}")
            
            # 如果config.data.num_labels看起来不合理（比如为2但可能是MNLI），尝试其他方法
            # 检查model_name或config中是否有其他线索
            if config_num_labels == 2:
                # 检查是否可能是MNLI（三分类任务但被错误设置为2）
                # 如果label_list应该存在但没有，这可能是MNLI任务
                if hasattr(config, 'data'):
                    # 尝试检查是否有其他MNLI相关的线索
                    model_name_str = str(model_name).lower()
                    if 'mnli' in model_name_str or (hasattr(config.data, 'type') and 'mnli' in str(config.data.type).lower()):
                        logger.warning(f"⚠ Suspicious: config.data.num_labels=2 but this might be MNLI task! Forcing num_labels=3")
                        expected_num_labels = 3
                    else:
                        expected_num_labels = config_num_labels
                else:
                    expected_num_labels = config_num_labels
            else:
                expected_num_labels = config_num_labels
        else:
            # 最后的默认值 - 对于MNLI任务，默认应该是3
            # 但如果没有其他信息，使用3作为默认值（因为MNLI是常见的三分类任务）
            expected_num_labels = 3
            logger.warning(f"⚠ Could not determine num_labels, using default: 3 (assuming MNLI or similar 3-class task)")
    
    # 验证并修复config.data.num_labels（如果它被错误修改）
    if hasattr(config, 'data'):
        if hasattr(config.data, 'num_labels') and config.data.num_labels != expected_num_labels:
            logger.warning(f"config.data.num_labels={config.data.num_labels} does not match expected={expected_num_labels}, correcting...")
            config.data.num_labels = expected_num_labels
        elif not hasattr(config.data, 'num_labels'):
            config.data.num_labels = expected_num_labels
            logger.info(f"Set config.data.num_labels={expected_num_labels}")
    
    kwargs['num_labels'] = expected_num_labels
    logger.info(f"Final num_labels for model loading: {expected_num_labels}")

    # 验证num_labels是否有效
    if kwargs['num_labels'] <= 0:
        raise ValueError(f"Invalid num_labels: {kwargs['num_labels']}. Must be > 0")

    # 检查是否是本地路径
    if os.path.exists(model_name):
        # 如果是本地路径，直接使用
        model_path = model_name
        logger.info(f"Loading model from local path: {model_path}")
    else:
        # 如果是Hugging Face Hub名称，使用原名称
        model_path = model_name
        logger.info(f"Loading model from Hugging Face Hub: {model_path}")

    # 首先加载配置并更新num_labels
    try:
        model_config = AutoConfig.from_pretrained(model_path, **{k: v for k, v in kwargs.items() if k != 'num_labels'})
        if hasattr(model_config, 'num_labels'):
            old_config_num_labels = model_config.num_labels
            if old_config_num_labels != kwargs['num_labels']:
                logger.info(f"Updating config.num_labels before loading: {old_config_num_labels} -> {kwargs['num_labels']}")
                model_config.num_labels = kwargs['num_labels']
        else:
            logger.info(f"Config does not have num_labels attribute, setting it to {kwargs['num_labels']}")
            model_config.num_labels = kwargs['num_labels']
        
        # 使用更新后的配置加载模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            config=model_config,
            **{k: v for k, v in kwargs.items() if k != 'num_labels'}
        )
    except Exception as e:
        logger.warning(f"Failed to pre-update config, falling back to standard loading: {e}")
        # 如果配置更新失败，回退到标准加载方式
        base_model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    
    # 验证并修复模型分类器输出维度
    expected_num_labels = kwargs['num_labels']
    
    # 立即设置模型的num_labels属性（对于某些模型类型很重要）
    if hasattr(base_model, 'num_labels'):
        old_num_labels = base_model.num_labels
        if old_num_labels != expected_num_labels:
            logger.info(f"Setting model.num_labels: {old_num_labels} -> {expected_num_labels}")
            base_model.num_labels = expected_num_labels
    
    classifier_fixed = False
    
    # 检查并修复分类器层
    for name, module in base_model.named_modules():
        if hasattr(module, 'out_features') and hasattr(module, 'in_features'):
            # 这是一个线性层，可能是分类器
            if 'classifier' in name.lower() or 'score' in name.lower() or 'head' in name.lower():
                if module.out_features != expected_num_labels:
                    logger.warning(f"Fixing classifier {name}: out_features {module.out_features} -> {expected_num_labels}")
                    # 创建新的分类器层
                    new_classifier = torch.nn.Linear(
                        module.in_features,
                        expected_num_labels,
                        bias=hasattr(module, 'bias') and module.bias is not None
                    )
                    # 如果维度兼容，尝试迁移权重（如果可能）
                    if module.out_features > expected_num_labels:
                        # 如果原来是更大的输出，取前expected_num_labels个
                        new_classifier.weight.data = module.weight.data[:expected_num_labels, :]
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_classifier.bias.data = module.bias.data[:expected_num_labels]
                    elif module.out_features < expected_num_labels:
                        # 如果原来是更小的输出，用零填充
                        new_classifier.weight.data[:module.out_features, :] = module.weight.data
                        if module.out_features < expected_num_labels:
                            # 初始化新权重的标准差
                            std = module.weight.data.std().item()
                            new_classifier.weight.data[module.out_features:, :].normal_(0, std * 0.01)
                        if hasattr(module, 'bias') and module.bias is not None:
                            new_classifier.bias.data[:module.out_features] = module.bias.data
                            new_classifier.bias.data[module.out_features:].zero_()
                    
                    # 替换模块
                    parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                    if parent_name:
                        parent_module = dict(base_model.named_modules())[parent_name]
                        setattr(parent_module, child_name, new_classifier)
                    else:
                        setattr(base_model, child_name, new_classifier)
                    classifier_fixed = True
                    logger.info(f"Successfully fixed classifier {name} to have out_features={expected_num_labels}")
                else:
                    logger.info(f"Classifier {name} correctly configured with out_features={module.out_features}")
                    classifier_fixed = True
    
    # 如果上面的方法没有找到分类器，尝试直接检查模型属性
    if not classifier_fixed:
        if hasattr(base_model, 'classifier'):
            classifier = base_model.classifier
            if isinstance(classifier, torch.nn.Linear):
                if classifier.out_features != expected_num_labels:
                    logger.warning(f"Fixing model.classifier: out_features {classifier.out_features} -> {expected_num_labels}")
                    in_features = classifier.in_features
                    has_bias = classifier.bias is not None
                    new_classifier = torch.nn.Linear(in_features, expected_num_labels, bias=has_bias)
                    
                    # 迁移权重
                    if classifier.out_features > expected_num_labels:
                        new_classifier.weight.data = classifier.weight.data[:expected_num_labels, :]
                        if has_bias:
                            new_classifier.bias.data = classifier.bias.data[:expected_num_labels]
                    elif classifier.out_features < expected_num_labels:
                        new_classifier.weight.data[:classifier.out_features, :] = classifier.weight.data
                        std = classifier.weight.data.std().item()
                        new_classifier.weight.data[classifier.out_features:, :].normal_(0, std * 0.01)
                        if has_bias:
                            new_classifier.bias.data[:classifier.out_features] = classifier.bias.data
                            new_classifier.bias.data[classifier.out_features:].zero_()
                    else:
                        new_classifier.weight.data = classifier.weight.data
                        if has_bias:
                            new_classifier.bias.data = classifier.bias.data
                    
                    base_model.classifier = new_classifier
                    logger.info(f"Successfully fixed model.classifier to have out_features={expected_num_labels}")
        elif hasattr(base_model, 'score'):
            score = base_model.score
            if isinstance(score, torch.nn.Linear):
                if score.out_features != expected_num_labels:
                    logger.warning(f"Fixing model.score: out_features {score.out_features} -> {expected_num_labels}")
                    in_features = score.in_features
                    has_bias = score.bias is not None
                    new_score = torch.nn.Linear(in_features, expected_num_labels, bias=has_bias)
                    
                    # 迁移权重
                    if score.out_features > expected_num_labels:
                        new_score.weight.data = score.weight.data[:expected_num_labels, :]
                        if has_bias:
                            new_score.bias.data = score.bias.data[:expected_num_labels]
                    elif score.out_features < expected_num_labels:
                        new_score.weight.data[:score.out_features, :] = score.weight.data
                        std = score.weight.data.std().item()
                        new_score.weight.data[score.out_features:, :].normal_(0, std * 0.01)
                        if has_bias:
                            new_score.bias.data[:score.out_features] = score.bias.data
                            new_score.bias.data[score.out_features:].zero_()
                    else:
                        new_score.weight.data = score.weight.data
                        if has_bias:
                            new_score.bias.data = score.bias.data
                    
                    base_model.score = new_score
                    logger.info(f"Successfully fixed model.score to have out_features={expected_num_labels}")
    
    # 最终验证：检查模型的config和num_labels属性
    if hasattr(base_model, 'config') and hasattr(base_model.config, 'num_labels'):
        if base_model.config.num_labels != expected_num_labels:
            logger.info(f"Updating model config num_labels: {base_model.config.num_labels} -> {expected_num_labels}")
            base_model.config.num_labels = expected_num_labels
    
    # 直接设置模型的num_labels属性（对于LLaMA等模型很重要）
    if hasattr(base_model, 'num_labels'):
        if base_model.num_labels != expected_num_labels:
            logger.info(f"Updating model.num_labels attribute: {base_model.num_labels} -> {expected_num_labels}")
            base_model.num_labels = expected_num_labels
    
    # 对于LLaMA模型，还需要检查model.score或model.classifier是否存在，并确保它们正确
    # 最终强制验证和修复分类器（确保万无一失）
    # 在最终检查前，重新从config中读取正确的num_labels（以防之前推断错误）
    # 优先级：label_list > dataset type > config.data.num_labels
    final_expected_num_labels = expected_num_labels
    if hasattr(config, 'data'):
        # 最高优先级：使用label_list（最可靠，即使type被错误修改）
        if hasattr(config.data, 'label_list') and config.data.label_list:
            label_list_len = len(config.data.label_list)
            if label_list_len > 0:
                final_expected_num_labels = label_list_len
                logger.info(f"Final check: Using label_list length ({label_list_len}): {final_expected_num_labels} (HIGHEST PRIORITY)")
        # 次优先级：从数据集类型推断
        elif hasattr(config.data, 'type'):
            data_type = str(config.data.type).lower()
            if 'mnli' in data_type:
                final_expected_num_labels = 3
                logger.info(f"Final check: Re-confirmed MNLI dataset from type, num_labels MUST be 3")
            elif 'qnli' in data_type or 'mrpc' in data_type or 'qqp' in data_type or 'sst' in data_type:
                final_expected_num_labels = 2
                logger.info(f"Final check: Re-confirmed binary classification from type, num_labels MUST be 2")
        # 最低优先级：使用config.data.num_labels
        if final_expected_num_labels == expected_num_labels and hasattr(config.data, 'num_labels'):
            final_expected_num_labels = config.data.num_labels
            logger.info(f"Final check: Using config.data.num_labels: {final_expected_num_labels}")
    
    # 如果最终期望值与当前值不一致，更新expected_num_labels
    if final_expected_num_labels != expected_num_labels:
        logger.warning(f"Final check: expected_num_labels mismatch! Previous: {expected_num_labels}, Corrected: {final_expected_num_labels}")
        expected_num_labels = final_expected_num_labels
        # 同时更新config和模型属性
        if hasattr(base_model, 'config'):
            base_model.config.num_labels = expected_num_labels
        if hasattr(base_model, 'num_labels'):
            base_model.num_labels = expected_num_labels
    
    if hasattr(base_model, 'score'):
        if isinstance(base_model.score, torch.nn.Linear):
            if base_model.score.out_features != expected_num_labels:
                logger.error(f"Final check FAILED: model.score.out_features={base_model.score.out_features} != {expected_num_labels}, FORCING FIX...")
                # 强制修复分类器
                in_features = base_model.score.in_features
                has_bias = base_model.score.bias is not None
                new_score = torch.nn.Linear(in_features, expected_num_labels, bias=has_bias)
                
                # 迁移权重
                if base_model.score.out_features < expected_num_labels:
                    new_score.weight.data[:base_model.score.out_features, :] = base_model.score.weight.data
                    std = base_model.score.weight.data.std().item()
                    new_score.weight.data[base_model.score.out_features:, :].normal_(0, std * 0.01)
                    if has_bias:
                        new_score.bias.data[:base_model.score.out_features] = base_model.score.bias.data
                        new_score.bias.data[base_model.score.out_features:].zero_()
                else:
                    new_score.weight.data = base_model.score.weight.data[:expected_num_labels, :]
                    if has_bias:
                        new_score.bias.data = base_model.score.bias.data[:expected_num_labels]
                
                base_model.score = new_score
                logger.info(f"Force fixed model.score to have out_features={expected_num_labels} ✓")
            else:
                logger.info(f"Final check: model.score.out_features={base_model.score.out_features} ✓")
    elif hasattr(base_model, 'classifier'):
        if isinstance(base_model.classifier, torch.nn.Linear):
            if base_model.classifier.out_features != expected_num_labels:
                logger.error(f"Final check FAILED: model.classifier.out_features={base_model.classifier.out_features} != {expected_num_labels}, FORCING FIX...")
                # 强制修复分类器
                in_features = base_model.classifier.in_features
                has_bias = base_model.classifier.bias is not None
                new_classifier = torch.nn.Linear(in_features, expected_num_labels, bias=has_bias)
                
                # 迁移权重
                if base_model.classifier.out_features < expected_num_labels:
                    new_classifier.weight.data[:base_model.classifier.out_features, :] = base_model.classifier.weight.data
                    std = base_model.classifier.weight.data.std().item()
                    new_classifier.weight.data[base_model.classifier.out_features:, :].normal_(0, std * 0.01)
                    if has_bias:
                        new_classifier.bias.data[:base_model.classifier.out_features] = base_model.classifier.bias.data
                        new_classifier.bias.data[base_model.classifier.out_features:].zero_()
                else:
                    new_classifier.weight.data = base_model.classifier.weight.data[:expected_num_labels, :]
                    if has_bias:
                        new_classifier.bias.data = base_model.classifier.bias.data[:expected_num_labels]
                
                base_model.classifier = new_classifier
                logger.info(f"Force fixed model.classifier to have out_features={expected_num_labels} ✓")
            else:
                logger.info(f"Final check: model.classifier.out_features={base_model.classifier.out_features} ✓")
    else:
        # 如果没有找到分类器，尝试从模型输出推断
        logger.warning("Could not find classifier layer for final verification")
    
    logger.info(f"Model loaded with num_labels={expected_num_labels}, config.num_labels={getattr(base_model.config, 'num_labels', 'N/A')}, model.num_labels={getattr(base_model, 'num_labels', 'N/A')}")
    return base_model


def get_dual_lora_llm(config, client_id=None):
    """
    创建双模块LoRA语言模型
    
    Args:
        config: 配置对象或模型配置对象
        client_id: 客户端ID
        
    Returns:
        双模块LoRA模型实例
    """
    # 检查config是否有model属性，如果没有则config本身就是model配置
    if hasattr(config, 'model'):
        model_config = config.model
        full_config = config
    else:
        model_config = config
        # 需要从全局获取完整配置
        from federatedscope.core.configs.config import global_cfg
        full_config = global_cfg
    
    # 获取模型类型信息
    model_type = getattr(model_config, 'type', None)
    if model_type is None:
        raise ValueError("Model type not found in config")
    
    # 检查 model_type 是否包含 '@' 分隔符
    if '@' not in model_type:
        raise ValueError(f"Invalid model type format: {model_type}. Expected format: 'model_name@hub'")
    
    model_name, model_hub = model_type.split('@')
    
    if model_hub == 'huggingface_llm':
        base_model = get_model_from_huggingface_dual_lora(model_name=model_name, config=full_config)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in {model_hub}.')

    # 获取适配器参数
    adapter_args = full_config.llm.adapter.args[0] if len(full_config.llm.adapter.args[0]) > 0 else {}
    
    # 双模块LoRA特定参数
    dual_lora_args = {
        'global_r': adapter_args.get('global_r', 8),
        'local_r': adapter_args.get('local_r', 4),
        'lora_alpha': adapter_args.get('lora_alpha', 16),
        'lora_dropout': adapter_args.get('lora_dropout', 0.05),
        'fusion_method': adapter_args.get('fusion_method', 'weighted_sum'),
        'target_modules': adapter_args.get('target_modules', ["query", "value", "key", "dense"]),
    }
    
    # 支持异构rank：根据客户端ID动态设置LoRA rank
    if (client_id is not None and 
        hasattr(full_config, 'aggregator') and 
        hasattr(full_config.aggregator, 'dual_lora_ranks') and
        full_config.aggregator.heter and
        str(client_id) in full_config.aggregator.dual_lora_ranks):
        
        # 使用客户端特定的rank配置
        client_ranks = full_config.aggregator.dual_lora_ranks[str(client_id)]
        if isinstance(client_ranks, (list, tuple)) and len(client_ranks) == 2:
            dual_lora_args['global_r'] = client_ranks[0]
            dual_lora_args['local_r'] = client_ranks[1]
            logger.info(f"客户端 {client_id} 使用双模块LoRA ranks: global_r={client_ranks[0]}, local_r={client_ranks[1]}")
    
    # 创建双模块LoRA模型
    model = DualLoraAdapterModel(
        model=base_model,
        use_adapter=full_config.llm.adapter.use,
        adapter_method='dual_lora',
        **dual_lora_args
    )
    
    # 确保num_labels在整个模型层次结构中正确设置
    # 使用与模型加载时相同的逻辑来获取正确的num_labels
    # 优先级：label_list > dataset type > config.data.num_labels
    expected_num_labels = None
    
    # 最高优先级：使用label_list（最可靠，即使type被错误修改）
    # 尝试多种方式访问label_list（与get_model_from_huggingface_dual_lora保持一致）
    label_list_value = None
    try:
        # 方式1：直接属性访问
        if hasattr(full_config.data, 'label_list'):
            label_list_value = getattr(full_config.data, 'label_list', None)
            logger.info(f"DEBUG: Found label_list via hasattr: {label_list_value} (type: {type(label_list_value)})")
    except Exception as e:
        logger.warning(f"DEBUG: Error accessing label_list via hasattr: {e}")
    
    # 如果方式1失败，尝试方式2：字典访问
    if not label_list_value:
        try:
            if isinstance(full_config.data, dict) or hasattr(full_config.data, '__getitem__'):
                label_list_value = full_config.data.get('label_list') if hasattr(full_config.data, 'get') else full_config.data['label_list']
                logger.info(f"DEBUG: Found label_list via dict access: {label_list_value}")
        except Exception as e:
            logger.warning(f"DEBUG: Error accessing label_list via dict: {e}")
    
    # 如果方式2失败，尝试方式3：通过__dict__访问
    if not label_list_value:
        try:
            if hasattr(full_config.data, '__dict__'):
                label_list_value = full_config.data.__dict__.get('label_list')
                logger.info(f"DEBUG: Found label_list via __dict__: {label_list_value}")
        except Exception as e:
            logger.warning(f"DEBUG: Error accessing label_list via __dict__: {e}")
    
    # 处理找到的label_list
    if label_list_value:
        try:
            label_list_num = len(label_list_value)
            logger.info(f"DEBUG: label_list length = {label_list_num}")
            if label_list_num > 0:
                expected_num_labels = label_list_num
                logger.info(f"✓ Inferred num_labels from label_list ({label_list_value}): {expected_num_labels} (HIGHEST PRIORITY)")
            else:
                logger.warning(f"DEBUG: label_list exists but is empty!")
        except Exception as e:
            logger.error(f"DEBUG: Error getting label_list length: {e}")
    else:
        logger.warning(f"DEBUG: Could not find label_list using any method in get_dual_lora_llm")
    
    # 次优先级：从数据集类型推断
    if expected_num_labels is None and hasattr(full_config.data, 'type'):
        data_type = str(full_config.data.type).lower()
        if 'mnli' in data_type:
            expected_num_labels = 3
            logger.info("Dataset type is MNLI, num_labels MUST be 3 (forcing override)")
        elif 'qnli' in data_type or 'mrpc' in data_type or 'qqp' in data_type or 'sst' in data_type:
            expected_num_labels = 2
            logger.info(f"Dataset type is {data_type}, num_labels MUST be 2 (forcing override)")
    
    # 最低优先级：从配置读取（但如果值是2且可能是MNLI，则怀疑并检查）
    if expected_num_labels is None:
        if hasattr(full_config.data, 'num_labels'):
            config_num_labels = full_config.data.num_labels
            logger.warning(f"Could not infer from label_list or dataset type, config.data.num_labels = {config_num_labels}")
            
            # 如果config.data.num_labels看起来不合理（比如为2但可能是MNLI），尝试其他方法
            if config_num_labels == 2:
                # 检查是否可能是MNLI（三分类任务但被错误设置为2）
                # 检查model_type、expname或其他配置中是否有MNLI相关的线索
                mnli_clues = []
                if hasattr(full_config.data, 'type') and 'mnli' in str(full_config.data.type).lower():
                    mnli_clues.append("config.data.type")
                if 'model_type' in locals() and 'mnli' in str(model_type).lower():
                    mnli_clues.append("model_type")
                if hasattr(full_config, 'expname') and 'mnli' in str(full_config.expname).lower():
                    mnli_clues.append("expname")
                
                if mnli_clues:
                    logger.warning(f"⚠ Suspicious: config.data.num_labels=2 but found MNLI clues in {', '.join(mnli_clues)}! Forcing num_labels=3")
                    expected_num_labels = 3
                else:
                    expected_num_labels = config_num_labels
            else:
                expected_num_labels = config_num_labels
        else:
            # 最后的默认值 - 使用3（因为MNLI是常见的三分类任务）
            expected_num_labels = 3
            logger.warning("Could not determine num_labels, using default: 3 (assuming MNLI or similar 3-class task)")
    
    # 确保config.data.num_labels是正确的
    if hasattr(full_config.data, 'num_labels') and full_config.data.num_labels != expected_num_labels:
        logger.warning(f"Correcting config.data.num_labels: {full_config.data.num_labels} -> {expected_num_labels}")
        full_config.data.num_labels = expected_num_labels
    elif not hasattr(full_config.data, 'num_labels'):
        full_config.data.num_labels = expected_num_labels
    
    logger.info(f"Using num_labels={expected_num_labels} for model configuration")
    
    def update_num_labels_recursive(obj, depth=0, max_depth=5):
        """递归更新模型所有层次的num_labels"""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        
        # 更新config.num_labels
        if hasattr(obj, 'config') and hasattr(obj.config, 'num_labels'):
            if obj.config.num_labels != expected_num_labels:
                logger.info(f"{indent}Updating {type(obj).__name__}.config.num_labels: {obj.config.num_labels} -> {expected_num_labels}")
                obj.config.num_labels = expected_num_labels
        
        # 更新num_labels属性
        if hasattr(obj, 'num_labels'):
            if obj.num_labels != expected_num_labels:
                logger.info(f"{indent}Updating {type(obj).__name__}.num_labels: {obj.num_labels} -> {expected_num_labels}")
                obj.num_labels = expected_num_labels
        
        # 递归更新子模型
        if hasattr(obj, 'model'):
            update_num_labels_recursive(obj.model, depth + 1, max_depth)
        if hasattr(obj, 'base_model'):
            update_num_labels_recursive(obj.base_model, depth + 1, max_depth)
    
    # 递归更新所有层次的num_labels
    logger.info("Updating num_labels in all model layers...")
    update_num_labels_recursive(model)
    
    # 最后验证：检查最终的底层模型
    current = model
    while hasattr(current, 'model'):
        current = current.model
        if hasattr(current, 'num_labels'):
            if current.num_labels != expected_num_labels:
                logger.error(f"Found model layer with incorrect num_labels: {type(current).__name__}.num_labels={current.num_labels}")
            else:
                logger.info(f"Verified {type(current).__name__}.num_labels={expected_num_labels} ✓")
        if hasattr(current, 'config') and hasattr(current.config, 'num_labels'):
            if current.config.num_labels != expected_num_labels:
                logger.error(f"Found model layer with incorrect config.num_labels: {type(current).__name__}.config.num_labels={current.config.num_labels}")
            else:
                logger.info(f"Verified {type(current).__name__}.config.num_labels={expected_num_labels} ✓")
    
    logger.info(f"Dual-LoRA model created with num_labels={expected_num_labels}")
    
    # 处理冻结策略
    if hasattr(full_config.federate, 'freeze_global') and full_config.federate.freeze_global:
        # 冻结全局适配器参数
        for name, param in model.named_parameters():
            if "global_lora" in name:
                param.requires_grad = False
        logger.info("Frozen global adapter parameters")
    
    if hasattr(full_config.federate, 'freeze_local') and full_config.federate.freeze_local:
        # 冻结本地适配器参数
        for name, param in model.named_parameters():
            if "local_lora" in name or "fusion" in name:
                param.requires_grad = False
        logger.info("Frozen local adapter parameters")
    
    # 处理全局适配器A矩阵冻结策略
    if hasattr(full_config.federate, 'freeze_A') and full_config.federate.freeze_A:
        # 冻结全局适配器的A矩阵参数
        frozen_count = 0
        for name, param in model.named_parameters():
            if "global_lora_A" in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Frozen {frozen_count} global adapter A matrices")
    else:
        # 确保全局适配器A矩阵参数可训练
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if "global_lora_A" in name:
                param.requires_grad = True
                unfrozen_count += 1
        if unfrozen_count > 0:
            logger.info(f"Ensured {unfrozen_count} global adapter A matrices are trainable")
    
    # 保存初始LoRA参数（用于本地训练）
    if full_config.federate.method == "local":
        initial_dual_lora_params = {
            name: param.clone() 
            for name, param in model.named_parameters() 
            if any(keyword in name for keyword in ['global_lora', 'local_lora', 'fusion'])
        }
        torch.save(initial_dual_lora_params, full_config.federate.save_to + '.dual_lora_init')
        logger.info(f"Saved initial dual-LoRA parameters to {full_config.federate.save_to}.dual_lora_init")
    
    return model


def create_dual_lora_model_from_config(config, client_id=None):
    """
    从配置创建双模块LoRA模型的便捷函数
    
    Args:
        config: 配置对象
        client_id: 客户端ID
        
    Returns:
        双模块LoRA模型
    """
    return get_dual_lora_llm(config, client_id)


class DualLoRAModelBuilder:
    """
    双模块LoRA模型构建器类
    提供更灵活的模型构建接口
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_model(self, client_id=None, **kwargs):
        """
        构建双模块LoRA模型
        
        Args:
            client_id: 客户端ID
            **kwargs: 额外参数
            
        Returns:
            双模块LoRA模型
        """
        return get_dual_lora_llm(self.config, client_id)
    
    def build_base_model(self):
        """
        构建基础模型（不应用LoRA）
        
        Returns:
            基础预训练模型
        """
        # 获取模型类型信息
        model_type = getattr(self.config.model, 'type', None)
        if model_type is None:
            raise ValueError("Model type not found in config")
        
        # 检查 model_type 是否包含 '@' 分隔符
        if '@' not in model_type:
            raise ValueError(f"Invalid model type format: {model_type}. Expected format: 'model_name@hub'")
        
        model_name, model_hub = model_type.split('@')
        
        if model_hub == 'huggingface_llm':
            return get_model_from_huggingface_dual_lora(model_name=model_name, config=self.config)
        else:
            raise NotImplementedError(f'Not support LLM {model_name} in {model_hub}.')
    
    def apply_dual_lora(self, base_model, client_id=None, **adapter_kwargs):
        """
        为基础模型应用双模块LoRA
        
        Args:
            base_model: 基础模型
            client_id: 客户端ID
            **adapter_kwargs: 适配器参数
            
        Returns:
            应用了双模块LoRA的模型
        """
        # 获取默认适配器参数
        default_args = self.config.llm.adapter.args[0] if len(self.config.llm.adapter.args[0]) > 0 else {}
        
        dual_lora_args = {
            'global_r': default_args.get('global_r', 8),
            'local_r': default_args.get('local_r', 4),
            'lora_alpha': default_args.get('lora_alpha', 16),
            'lora_dropout': default_args.get('lora_dropout', 0.05),
            'fusion_method': default_args.get('fusion_method', 'weighted_sum'),
            'target_modules': default_args.get('target_modules', ["query", "value", "key", "dense"]),
        }
        
        # 更新参数
        dual_lora_args.update(adapter_kwargs)
        
        # 异构配置
        if (client_id is not None and 
            hasattr(self.config, 'aggregator') and 
            hasattr(self.config.aggregator, 'dual_lora_ranks') and
            self.config.aggregator.heter and
            str(client_id) in self.config.aggregator.dual_lora_ranks):
            
            client_ranks = self.config.aggregator.dual_lora_ranks[str(client_id)]
            if isinstance(client_ranks, (list, tuple)) and len(client_ranks) == 2:
                dual_lora_args['global_r'] = client_ranks[0]
                dual_lora_args['local_r'] = client_ranks[1]
        
        # 创建双模块LoRA配置
        config = DualLoraConfig(**dual_lora_args)
        
        # 应用双模块LoRA
        return enable_dual_lora_adapter(base_model, **dual_lora_args)
    
    def get_client_specific_config(self, client_id):
        """
        获取客户端特定的配置
        
        Args:
            client_id: 客户端ID
            
        Returns:
            客户端特定的配置字典
        """
        client_config = {}
        
        # 异构rank配置
        if (hasattr(self.config, 'aggregator') and 
            hasattr(self.config.aggregator, 'dual_lora_ranks') and
            self.config.aggregator.heter and
            str(client_id) in self.config.aggregator.dual_lora_ranks):
            
            client_ranks = self.config.aggregator.dual_lora_ranks[str(client_id)]
            if isinstance(client_ranks, (list, tuple)) and len(client_ranks) == 2:
                client_config['global_r'] = client_ranks[0]
                client_config['local_r'] = client_ranks[1]
        
        return client_config
    
    def print_model_info(self, model):
        """
        打印模型信息
        
        Args:
            model: 模型实例
        """
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")


# 兼容性函数，保持与原有接口一致
def get_llm(config, client_id=None):
    """
    兼容原有接口的模型获取函数
    
    Args:
        config: 配置对象
        client_id: 客户端ID
        
    Returns:
        模型实例（根据配置决定是否使用双模块LoRA）
    """
    # 检查是否启用双模块LoRA
    adapter_args = config.llm.adapter.args[0] if len(config.llm.adapter.args[0]) > 0 else {}
    use_dual_lora = adapter_args.get('use_dual_lora', False)
    
    if use_dual_lora:
        # 使用双模块LoRA
        return get_dual_lora_llm(config, client_id)
    else:
        # 回退到原有实现
        from federatedscope.glue.model.model_builder import get_llm as original_get_llm
        return original_get_llm(config, client_id)


# 工厂函数
def create_model_builder(config):
    """
    创建模型构建器的工厂函数
    
    Args:
        config: 配置对象
        
    Returns:
        DualLoRAModelBuilder实例
    """
    return DualLoRAModelBuilder(config)


# 模型验证函数
def validate_dual_lora_model(model):
    """
    验证双模块LoRA模型的正确性
    
    Args:
        model: 模型实例
        
    Returns:
        bool: 验证是否通过
    """
    try:
        # 检查是否有双模块LoRA层
        has_dual_lora = False
        for name, module in model.named_modules():
            if hasattr(module, 'global_lora_A') and hasattr(module, 'local_lora_A'):
                has_dual_lora = True
                break
        
        if not has_dual_lora:
            logger.warning("Model does not contain dual-LoRA layers")
            return False
        
        # 检查参数是否可训练
        global_trainable = any(
            param.requires_grad for name, param in model.named_parameters()
            if 'global_lora' in name
        )
        
        local_trainable = any(
            param.requires_grad for name, param in model.named_parameters()
            if 'local_lora' in name
        )
        
        if not (global_trainable or local_trainable):
            logger.warning("No dual-LoRA parameters are trainable")
            return False
        
        logger.info("Dual-LoRA model validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Dual-LoRA model validation failed: {e}")
        return False


# 模型转换函数
def convert_single_to_dual_lora(single_lora_model, config, client_id=None):
    """
    将单模块LoRA模型转换为双模块LoRA模型
    
    Args:
        single_lora_model: 单模块LoRA模型
        config: 配置对象
        client_id: 客户端ID
        
    Returns:
        双模块LoRA模型
    """
    logger.info("Converting single-LoRA model to dual-LoRA model")
    
    # 获取基础模型
    if hasattr(single_lora_model, 'model'):
        base_model = single_lora_model.model
    else:
        base_model = single_lora_model
    
    # 创建双模块LoRA模型
    builder = DualLoRAModelBuilder(config)
    dual_lora_model = builder.apply_dual_lora(base_model, client_id)
    
    # 尝试迁移参数（如果可能）
    try:
        single_state = single_lora_model.state_dict()
        dual_state = dual_lora_model.state_dict()
        
        # 迁移兼容的参数
        for key in single_state:
            if key in dual_state and single_state[key].shape == dual_state[key].shape:
                dual_state[key] = single_state[key]
        
        dual_lora_model.load_state_dict(dual_state, strict=False)
        logger.info("Successfully migrated compatible parameters")
        
    except Exception as e:
        logger.warning(f"Parameter migration failed: {e}")
    
    return dual_lora_model
