#!/usr/bin/env python3
"""
MMLU评估示例脚本
演示如何使用MMLU评估功能

python example_mmlu_evaluation.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_quick_evaluation():
    """运行快速评估示例"""
    logger.info("Running quick evaluation example...")
    
    # 检查是否有可用的模型检查点
    model_paths = [
        "/home/szk_25/FedSA-LoRA-Dual/exp/dual-lora_/home/szk_25/FederatedLLM/llama-7b@huggingface_llm_on_sst2@glue_lr0.0002_lstep10/sub_exp_20251012235324",
        "/home/szk_25/FedSA-LoRA-Dual/exp/dual-lora_/home/szk_25/FederatedLLM/llama-7b@huggingface_llm_on_sst2@glue_lr0.0002_lstep10/sub_exp_20251007193207",
        "/home/szk_25/FedSA-LoRA-Dual/exp/dual-lora_/home/szk_25/FederatedLLM/llama-7b@huggingface_llm_on_sst2@glue_lr0.0002_lstep10/sub_exp_20251012231038"
    ]
    
    available_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            available_model = model_path
            break
    
    if not available_model:
        logger.warning("No trained model checkpoints found. Using base model for demonstration.")
        available_model = "/home/szk_25/FederatedLLM/llama-7b@huggingface_llm"
    
    logger.info(f"Using model: {available_model}")
    
    # 运行快速评估
    cmd = [
        'python', 'quick_mmlu_eval.py',
        '--model_path', available_model,
        '--output_dir', './example_results'
    ]
    
    try:
        logger.info("Executing quick evaluation...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30分钟超时
        
        if result.returncode == 0:
            logger.info("Quick evaluation completed successfully!")
            logger.info("Output:")
            print(result.stdout)
        else:
            logger.error(f"Quick evaluation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Quick evaluation timed out")
    except Exception as e:
        logger.error(f"Quick evaluation failed with exception: {e}")

def run_detailed_evaluation():
    """运行详细评估示例"""
    logger.info("Running detailed evaluation example...")
    
    # 检查配置文件
    config_path = "/home/szk_25/FedSA-LoRA-Dual/mmlu_evaluation_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    # 运行详细评估
    cmd = [
        'python', 'mmlu_evaluator.py',
        '--config', config_path,
        '--output_dir', './example_detailed_results',
        '--verbose'
    ]
    
    try:
        logger.info("Executing detailed evaluation...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        if result.returncode == 0:
            logger.info("Detailed evaluation completed successfully!")
            logger.info("Output:")
            print(result.stdout)
        else:
            logger.error(f"Detailed evaluation failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Detailed evaluation timed out")
    except Exception as e:
        logger.error(f"Detailed evaluation failed with exception: {e}")

def run_variant_comparison():
    """运行变体对比示例"""
    logger.info("Running variant comparison example...")
    
    # 检查配置文件
    config_path = "/home/szk_25/FedSA-LoRA-Dual/mmlu_evaluation_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    # 运行变体对比
    cmd = [
        'python', 'run_mmlu_experiments.py',
        '--config', config_path,
        '--variants', 'global_only,dual_fusion',
        '--output_dir', './example_variant_results'
    ]
    
    try:
        logger.info("Executing variant comparison...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        if result.returncode == 0:
            logger.info("Variant comparison completed successfully!")
            logger.info("Output:")
            print(result.stdout)
        else:
            logger.error(f"Variant comparison failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("Variant comparison timed out")
    except Exception as e:
        logger.error(f"Variant comparison failed with exception: {e}")

def main():
    """主函数"""
    logger.info("Starting MMLU evaluation examples...")
    
    # 创建示例输出目录
    os.makedirs('./example_results', exist_ok=True)
    os.makedirs('./example_detailed_results', exist_ok=True)
    os.makedirs('./example_variant_results', exist_ok=True)
    
    # 选择要运行的示例
    print("\n选择要运行的示例:")
    print("1. 快速评估 (Quick Evaluation)")
    print("2. 详细评估 (Detailed Evaluation)")
    print("3. 变体对比 (Variant Comparison)")
    print("4. 运行所有示例 (Run All)")
    print("5. 退出 (Exit)")
    
    choice = input("\n请输入选择 (1-5): ").strip()
    
    if choice == '1':
        run_quick_evaluation()
    elif choice == '2':
        run_detailed_evaluation()
    elif choice == '3':
        run_variant_comparison()
    elif choice == '4':
        logger.info("Running all examples...")
        run_quick_evaluation()
        run_detailed_evaluation()
        run_variant_comparison()
    elif choice == '5':
        logger.info("Exiting...")
        return
    else:
        logger.error("Invalid choice. Please run the script again.")
        return
    
    logger.info("Example execution completed!")

if __name__ == '__main__':
    main()
