#!/usr/bin/env python3
"""
快速MMLU评估脚本
基于FederatedLLM的global_evaluation函数，适配双模块LoRA模型

使用方法:
python quick_mmlu_eval.py --model_path /path/to/model --data_path /path/to/data.jsonl
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
CURRENT_PROJECT_PATH = "/home/szk_25/FedSA-LoRA-Dual"
FEDERATEDLLM_PATH = "/home/szk_25/FederatedLLM"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)
if FEDERATEDLLM_PATH not in sys.path:
    sys.path.insert(0, FEDERATEDLLM_PATH)

# 导入必要模块
from fed_utils.evaluation import global_evaluation
from utils.prompter import Prompter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dual_lora_model(model_path: str):
    """加载双模块LoRA模型"""
    logger.info(f"Loading model from {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/szk_25/FederatedLLM/llama-7b@huggingface_llm",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        "/home/szk_25/FederatedLLM/llama-7b@huggingface_llm",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 尝试加载适配器权重
    adapter_weights_path = os.path.join(model_path, "adapter_model.bin")
    if os.path.exists(adapter_weights_path):
        logger.info(f"Loading adapter weights from {adapter_weights_path}")
        adapter_weights = torch.load(adapter_weights_path, map_location='cpu')
        model.load_state_dict(adapter_weights, strict=False)
    else:
        logger.warning(f"No adapter weights found at {adapter_weights_path}")
    
    return model, tokenizer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Quick MMLU Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                       default='/home/szk_25/FederatedLLM/mmlu_test_1444.jsonl',
                       help='Path to MMLU test data')
    parser.add_argument('--output_dir', type=str, default='./quick_mmlu_results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model, tokenizer = load_dual_lora_model(args.model_path)
    
    # 创建提示器
    prompter = Prompter("alpaca")
    
    # 运行评估
    logger.info("Starting MMLU evaluation...")
    accuracy = global_evaluation(model, tokenizer, prompter, args.data_path)
    
    # 保存结果
    results = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'accuracy': accuracy,
        'timestamp': str(torch.cuda.Event(enable_timing=True).record())
    }
    
    import json
    results_file = os.path.join(args.output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    logger.info(f"Results saved to {results_file}")

if __name__ == '__main__':
    main()
