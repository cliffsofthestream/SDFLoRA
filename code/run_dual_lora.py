#!/usr/bin/env python3
"""
双模块LoRA联邦学习运行脚本
参考: NeurIPS 2024 - Dual-Personalizing Adapter for Federated Foundation Models

setup:
python run_dual_lora.py --cfg dual_lora_config.yaml
python run_dual_lora.py --cfg dual_lora_hetero_config.yaml

DP-SGD support:
python run_dual_lora.py --cfg dual_lora_config.yaml --enable-dp-sgd
python run_dual_lora.py --cfg dual_lora_config.yaml --enable-dp-sgd --dp-epsilon 1.0 --dp-delta 1e-5
python run_dual_lora.py --cfg dual_lora_config.yaml --enable-dp-sgd --dp-epsilon 0.1 --dp-max-grad-norm 0.5
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加原项目路径到sys.path
ORIGINAL_PROJECT_PATH = "/home/szk_25/FedSA-LoRA"
if ORIGINAL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_PATH)

# 添加当前项目路径
CURRENT_PROJECT_PATH = "/home/szk_25/FedSA-LoRA-Dual"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 导入原项目模块
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, get_ds_rank
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner

# 导入双模块LoRA模块
from code.dual_lora_model_builder import get_dual_lora_llm, DualLoRAModelBuilder
from code.dual_lora_aggregator import DualLoRAFederatedAggregator
from code.dual_lora_peft_adapter import DualLoraAdapterModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_dual_lora_components():
    """注册双模块LoRA组件到FederatedScope"""
    try:
        # 动态添加聚合器类型到AGGREGATOR_TYPE字典
        from federatedscope.core.configs import constants
        constants.AGGREGATOR_TYPE['dual-lora'] = 'dual_lora_aggregator'
        logger.info("Added dual-lora to AGGREGATOR_TYPE")
        
        # 修改聚合器构建器以支持我们的自定义聚合器
        from federatedscope.core.auxiliaries import aggregator_builder
        
        # 保存原始的get_aggregator函数
        original_get_aggregator = aggregator_builder.get_aggregator
        
        def patched_get_aggregator(method, model=None, device=None, online=False, config=None):
            """修补后的聚合器构建器"""
            if method == 'dual-lora' or (config and hasattr(config, 'aggregator') and 
                                        hasattr(config.aggregator, 'type') and 
                                        config.aggregator.type == 'dual_lora_aggregator'):
                logger.info("Using DualLoRA Federated Aggregator")
                return DualLoRAFederatedAggregator(model=model, config=config, device=device)
            else:
                return original_get_aggregator(method, model, device, online, config)
        
        # 替换get_aggregator函数
        aggregator_builder.get_aggregator = patched_get_aggregator
        logger.info("Patched aggregator builder to support DualLoRA")
        
        # 注册模型构建器
        from federatedscope.register import register_model
        register_model('dual_lora_llm', get_dual_lora_llm)
        logger.info("Registered dual_lora_llm model builder")
        
    except ImportError as e:
        logger.warning(f"Could not register components with FederatedScope: {e}")
        logger.info("Will use standalone mode")


def setup_dual_lora_config(cfg):
    """设置双模块LoRA特定配置"""
    
    # 确保聚合器配置存在
    if not hasattr(cfg, 'aggregator'):
        cfg.aggregator = CfgNode()
    
    # 设置默认聚合器类型
    if not hasattr(cfg.aggregator, 'type'):
        cfg.aggregator.type = 'dual_lora_aggregator'
    
    # 设置双模块LoRA特定配置
    dual_lora_defaults = {
        'global_aggregation_strategy': 'fedavg',
        'local_personalization_strategy': 'local_only',
        'stacking': True,
        'zero_padding': False,
        'heter': True,
        'dual_lora_ranks': {}
    }
    
    for key, default_value in dual_lora_defaults.items():
        if not hasattr(cfg.aggregator, key):
            setattr(cfg.aggregator, key, default_value)
    
    # 如果启用了DP-SGD，设置聚合器的DP-SGD配置
    if hasattr(cfg, 'differential_privacy') and cfg.differential_privacy.enabled:
        logger.info("Setting up DP-SGD configuration for aggregator...")
        cfg.aggregator.enable_dp_sgd = True
        
        # 创建DP-SGD配置节点
        if not hasattr(cfg.aggregator, 'dp_config'):
            cfg.aggregator.dp_config = CfgNode()
        
        cfg.aggregator.dp_config.enabled = True
        cfg.aggregator.dp_config.epsilon = cfg.differential_privacy.epsilon
        cfg.aggregator.dp_config.delta = cfg.differential_privacy.delta
        cfg.aggregator.dp_config.max_grad_norm = cfg.differential_privacy.max_grad_norm
        cfg.aggregator.dp_config.apply_to_global = cfg.differential_privacy.apply_to_global
        cfg.aggregator.dp_config.apply_to_local = cfg.differential_privacy.apply_to_local
        cfg.aggregator.dp_config.global_noise_scale = cfg.differential_privacy.global_noise_scale
        cfg.aggregator.dp_config.local_noise_scale = cfg.differential_privacy.local_noise_scale
        cfg.aggregator.dp_config.enable_secure_aggregation = cfg.differential_privacy.enable_secure_aggregation
        cfg.aggregator.dp_config.aggregation_noise_scale = cfg.differential_privacy.aggregation_noise_scale
        
        logger.info("DP-SGD configuration applied to aggregator")
    
    # 确保个性化配置正确
    if not hasattr(cfg, 'personalization'):
        cfg.personalization = CfgNode()
    
    if not hasattr(cfg.personalization, 'local_param'):
        cfg.personalization.local_param = [
            'local_lora_A', 'local_lora_B', 
            'global_weight', 'local_weight', 'gate', 
            'classifier'
        ]
    
    # 确保LLM适配器配置
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter'):
        if len(cfg.llm.adapter.args) > 0:
            adapter_args = cfg.llm.adapter.args[0]
            if isinstance(adapter_args, dict):
                # 如果是字典，直接设置键值
                if 'use_dual_lora' not in adapter_args:
                    adapter_args['use_dual_lora'] = True
            else:
                # 如果是对象，设置属性
                if not hasattr(adapter_args, 'use_dual_lora'):
                    adapter_args.use_dual_lora = True
    
    logger.info("Dual-LoRA configuration setup completed")
    return cfg


def create_dual_lora_runner(data, config, client_configs=None):
    """创建双模块LoRA运行器"""
    
    # 获取服务器和客户端类
    server_class = get_server_cls(config)
    client_class = get_client_cls(config)
    
    # 创建运行器
    runner = get_runner(
        data=data,
        server_class=server_class,
        client_class=client_class,
        config=config.clone(),
        client_configs=client_configs
    )
    
    return runner


def main():
    """主函数"""
    # 先解析DP-SGD参数
    dp_args = argparse.Namespace()
    dp_args.enable_dp_sgd = False
    dp_args.dp_epsilon = 1.0
    dp_args.dp_delta = 1e-5
    dp_args.dp_max_grad_norm = 1.0
    
    # 手动解析DP-SGD参数并过滤掉它们
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--enable-dp-sgd':
            dp_args.enable_dp_sgd = True
        elif sys.argv[i] == '--dp-epsilon' and i + 1 < len(sys.argv):
            dp_args.dp_epsilon = float(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == '--dp-delta' and i + 1 < len(sys.argv):
            dp_args.dp_delta = float(sys.argv[i + 1])
            i += 1
        elif sys.argv[i] == '--dp-max-grad-norm' and i + 1 < len(sys.argv):
            dp_args.dp_max_grad_norm = float(sys.argv[i + 1])
            i += 1
        else:
            filtered_argv.append(sys.argv[i])
        i += 1
    
    # 临时替换sys.argv以过滤DP-SGD参数
    original_argv = sys.argv.copy()
    sys.argv = filtered_argv
    
    # 解析FederatedScope参数
    init_cfg = global_cfg.clone()
    args = parse_args()
    
    # 恢复原始sys.argv
    sys.argv = original_argv
    
    if args.cfg_file:
        # 先加载配置文件，但跳过differential_privacy部分
        import yaml
        with open(args.cfg_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 移除differential_privacy配置以避免FederatedScope错误
        if 'differential_privacy' in config_dict:
            dp_config_from_file = config_dict.pop('differential_privacy')
        else:
            dp_config_from_file = None
            
        # 移除aggregator中的自定义配置以避免FederatedScope错误
        aggregator_custom_config = None
        if 'aggregator' in config_dict:
            aggregator_config = config_dict['aggregator']
            # 保存自定义配置
            aggregator_custom_config = {
                'client_scaling_factor': aggregator_config.pop('client_scaling_factor', 0.1),
                'use_fixed_scaling': aggregator_config.pop('use_fixed_scaling', False)
            }
        
        # 将修改后的配置写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            yaml.dump(config_dict, tmp_file)
            tmp_config_file = tmp_file.name
        
        # 使用临时配置文件
        init_cfg.merge_from_file(tmp_config_file)
        
        # 清理临时文件
        import os
        os.unlink(tmp_config_file)
    
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)
    
    # 如果启用DP-SGD，设置相关配置
    if dp_args.enable_dp_sgd:
        logger.info("DP-SGD enabled with parameters:")
        logger.info(f"  Epsilon: {dp_args.dp_epsilon}")
        logger.info(f"  Delta: {dp_args.dp_delta}")
        logger.info(f"  Max grad norm: {dp_args.dp_max_grad_norm}")
        
        # 设置DP-SGD配置
        if not hasattr(init_cfg, 'differential_privacy'):
            init_cfg.differential_privacy = CfgNode()
        
        init_cfg.differential_privacy.enabled = True
        init_cfg.differential_privacy.epsilon = dp_args.dp_epsilon
        init_cfg.differential_privacy.delta = dp_args.dp_delta
        init_cfg.differential_privacy.max_grad_norm = dp_args.dp_max_grad_norm
        init_cfg.differential_privacy.apply_to_global = True
        init_cfg.differential_privacy.apply_to_local = False
        init_cfg.differential_privacy.global_noise_scale = 1.0
        init_cfg.differential_privacy.local_noise_scale = 0.5
        init_cfg.differential_privacy.enable_secure_aggregation = True
        init_cfg.differential_privacy.aggregation_noise_scale = 0.8
    elif dp_config_from_file and dp_config_from_file.get('enabled', False):
        # 如果配置文件中启用了DP-SGD，使用文件中的配置
        logger.info("DP-SGD enabled from config file")
        if not hasattr(init_cfg, 'differential_privacy'):
            init_cfg.differential_privacy = CfgNode()
        
        for key, value in dp_config_from_file.items():
            setattr(init_cfg.differential_privacy, key, value)
    
    # 设置双模块LoRA配置
    init_cfg = setup_dual_lora_config(init_cfg)
    
    # 添加自定义聚合器配置
    if aggregator_custom_config:
        logger.info("Adding custom aggregator configuration:")
        logger.info(f"  Client scaling factor: {aggregator_custom_config['client_scaling_factor']}")
        logger.info(f"  Use fixed scaling: {aggregator_custom_config['use_fixed_scaling']}")
        
        # 将自定义配置添加到聚合器配置中
        if not hasattr(init_cfg.aggregator, 'client_scaling_factor'):
            init_cfg.aggregator.client_scaling_factor = aggregator_custom_config['client_scaling_factor']
        if not hasattr(init_cfg.aggregator, 'use_fixed_scaling'):
            init_cfg.aggregator.use_fixed_scaling = aggregator_custom_config['use_fixed_scaling']
    
    # 注册双模块LoRA组件
    register_dual_lora_components()
    
    # DeepSpeed支持
    if init_cfg.llm.deepspeed.use:
        import deepspeed
        deepspeed.init_distributed()
    
    # 设置日志和随机种子
    update_logger(init_cfg, clear_before_add=True, rank=get_ds_rank())
    setup_seed(init_cfg.seed)
    
    # 加载客户端配置
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None
    
    # 获取数据
    logger.info("Loading data...")
    data, modified_cfg = get_data(config=init_cfg.clone(), client_cfgs=client_cfgs)
    
    # 处理配置合并问题 - 移除可能导致类型不匹配的字段
    if hasattr(modified_cfg, 'aggregator') and hasattr(modified_cfg.aggregator, 'dual_lora_ranks'):
        if isinstance(modified_cfg.aggregator.dual_lora_ranks, dict):
            # 如果dual_lora_ranks是字典，将其转换为配置对象或移除
            delattr(modified_cfg.aggregator, 'dual_lora_ranks')
    
    # 如果启用了DP-SGD，移除聚合器中的DP-SGD配置以避免合并冲突
    if hasattr(init_cfg, 'differential_privacy') and init_cfg.differential_privacy.enabled:
        if hasattr(modified_cfg, 'aggregator') and hasattr(modified_cfg.aggregator, 'dp_config'):
            delattr(modified_cfg.aggregator, 'dp_config')
        if hasattr(modified_cfg.aggregator, 'enable_dp_sgd'):
            delattr(modified_cfg.aggregator, 'enable_dp_sgd')
        # 也移除differential_privacy配置以避免合并冲突
        if hasattr(modified_cfg, 'differential_privacy'):
            delattr(modified_cfg, 'differential_privacy')
    
    init_cfg.merge_from_other_cfg(modified_cfg)
    
    # 处理本地训练配置
    if init_cfg.federate.client_idx_for_local_train != 0:
        init_cfg.federate.client_num = 1
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train]
        data = new_data
    
    # 冻结配置
    init_cfg.freeze()
    
    # 创建并运行双模块LoRA训练
    logger.info("Creating dual-LoRA runner...")
    runner = create_dual_lora_runner(
        data=data,
        config=init_cfg.clone(),
        client_configs=client_cfgs
    )
    
    logger.info("Starting dual-LoRA federated training...")
    result = runner.run()
    
    logger.info("Dual-LoRA federated training completed!")
    return result


def run_with_config_file(config_file):
    """使用配置文件运行"""
    # 设置命令行参数
    sys.argv = ['run_dual_lora.py', '--cfg', config_file]
    return main()


def demo_dual_lora():
    """双模块LoRA演示"""
    logger.info("Running Dual-LoRA Demo...")
    
    # 使用基础配置运行
    config_file = os.path.join(CURRENT_PROJECT_PATH, "dual_lora_config.yaml")
    if os.path.exists(config_file):
        logger.info(f"Running with config: {config_file}")
        result = run_with_config_file(config_file)
        logger.info("Demo completed successfully!")
        return result
    else:
        logger.error(f"Config file not found: {config_file}")
        return None


def demo_dual_lora_hetero():
    """双模块LoRA异构演示"""
    logger.info("Running Dual-LoRA Heterogeneous Demo...")
    
    # 使用异构配置运行
    config_file = os.path.join(CURRENT_PROJECT_PATH, "dual_lora_hetero_config.yaml")
    if os.path.exists(config_file):
        logger.info(f"Running with heterogeneous config: {config_file}")
        result = run_with_config_file(config_file)
        logger.info("Heterogeneous demo completed successfully!")
        return result
    else:
        logger.error(f"Config file not found: {config_file}")
        return None


if __name__ == '__main__':
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_dual_lora()
        elif sys.argv[1] == '--demo-hetero':
            demo_dual_lora_hetero()
        else:
            main()
    else:
        # 默认运行演示
        logger.info("No arguments provided, running demo...")
        demo_dual_lora()
