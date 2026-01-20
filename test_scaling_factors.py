#!/usr/bin/env python3

import argparse
import os
import sys
import yaml
import subprocess
from pathlib import Path

def create_config_with_scaling_factor(scaling_factor: float, use_fixed_scaling: bool = True):
    """创建带有指定缩放因子的配置文件"""
    
    # 基础配置
    base_config = {
        'seed': 789,
        'use_gpu': True,
        'device': 0,
        'early_stop': {'patience': 0},
        
        'differential_privacy': {
            'enabled': True,
            'epsilon': 2.0,
            'delta': 1e-4,
            'max_grad_norm': 1.0,
            'noise_multiplier': 1.1,
            'apply_to_global': True,
            'apply_to_local': False,
            'global_noise_scale': 1.0,
            'local_noise_scale': 0.5,
            'enable_secure_aggregation': True,
            'aggregation_noise_scale': 0.8
        },
        
        'federate': {
            'freeze_A': False,
            'mode': 'standalone',
            'client_num': 8,
            'total_round_num': 12,
            'save_to': f'dual_lora_mnli_pk{scaling_factor}.ckpt',
            'share_local_model': True,
            'online_aggr': False,
            'method': 'dual-lora'
        },
        
        'personalization': {
            'local_param': ['local_lora_A', 'local_lora_B', 'global_weight', 'local_weight', 'gate', 'classifier']
        },
        
        'aggregator': {
            'robust_rule': 'dual_lora_stacked',
            'stacking': True,
            'zero_padding': False,
            'heter': True,
            'client_scaling_factor': scaling_factor,
            'use_fixed_scaling': use_fixed_scaling,
            'local_ranks': {
                "0": 8, "1": 8, "2": 8, "3": 8,
                "4": 8, "5": 8, "6": 8, "7": 8
            }
        },
        
        'data': {
            'root': '/home/szk_25/FedSA-LoRA-Dual/GLUE',
            'type': 'mnli@glue',
            'matched': True,
            'splitter': 'lda',
            'splitter_args': [{'alpha': 0.5}]
        },
        
        'llm': {
            'tok_len': 128,
            'adapter': {
                'use': True,
                'args': [{
                    'adapter_package': 'dual_lora',
                    'adapter_method': 'dual_lora',
                    'use_dual_lora': True,
                    'global_r': 8,
                    'lora_alpha': 16,
                    'lora_dropout': 0.05,
                    'local_r': 4,
                    'fusion_method': 'weighted_sum',
                    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
                }]
            }
        },
        
        'dataloader': {'batch_size': 8},
        'model': {'type': '/home/szk_25/FederatedLLM/llama-7b@huggingface_llm'},
        'train': {
            'local_update_steps': 10,
            'batch_or_epoch': 'batch',
            'optimizer': {'lr': 2e-4},
            'is_enable_half': True
        },
        'criterion': {'type': 'CrossEntropyLoss'},
        'trainer': {'type': 'gluetrainer'},
        'eval': {
            'freq': 1,
            'metrics': ['accuracy', 'f1'],
            'count_flops': False,
            'best_res_update_round_wise_key': 'val_accuracy'
        }
    }
    
    return base_config

def run_experiment(scaling_factor: float, use_fixed_scaling: bool = True):
    """运行单个缩放因子实验"""
    
    print(f"\n{'='*60}")
    print(f"Testing client scaling factor (pk) = {scaling_factor}")
    print(f"Use fixed scaling: {use_fixed_scaling}")
    print(f"{'='*60}")
    
    # 创建配置文件
    config = create_config_with_scaling_factor(scaling_factor, use_fixed_scaling)
    config_file = f"dual_lora_config_pk{scaling_factor}.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config file: {config_file}")
    
    # 运行实验
    cmd = [
        'python', 'run_dual_lora.py',
        '--config', config_file,
        '--exp_name', f'dual_lora_pk{scaling_factor}',
        '--outdir', f'exp/dual_lora_pk{scaling_factor}'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        if result.returncode == 0:
            print(f"✅ Experiment completed successfully for pk={scaling_factor}")
        else:
            print(f"❌ Experiment failed for pk={scaling_factor}")
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment timed out for pk={scaling_factor}")
    except Exception as e:
        print(f"💥 Exception occurred for pk={scaling_factor}: {e}")
    
    # 清理配置文件
    if os.path.exists(config_file):
        os.remove(config_file)

def main():
    parser = argparse.ArgumentParser(description='Test different client scaling factors (pk)')
    parser.add_argument('--scaling_factors', nargs='+', type=float, 
                       default=[0.01, 0.05, 0.1, 0.2],
                       help='List of scaling factors to test')
    parser.add_argument('--use_fixed_scaling', action='store_true', default=True,
                       help='Use fixed scaling factor instead of data-size based weights')
    parser.add_argument('--single_test', type=float, default=None,
                       help='Test a single scaling factor value')
    
    args = parser.parse_args()
    
    if args.single_test is not None:
        # 测试单个缩放因子
        run_experiment(args.single_test, args.use_fixed_scaling)
    else:
        # 测试多个缩放因子
        print("Testing multiple client scaling factors (pk):")
        print(f"Values: {args.scaling_factors}")
        print(f"Use fixed scaling: {args.use_fixed_scaling}")
        
        for scaling_factor in args.scaling_factors:
            run_experiment(scaling_factor, args.use_fixed_scaling)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Check the exp/ directories for results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
