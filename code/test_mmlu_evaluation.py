#!/usr/bin/env python3
"""
set up:
python test_mmlu_evaluation.py
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
CURRENT_PROJECT_PATH = "/home/szk_25/FedSA-LoRA-Dual"
FEDERATEDLLM_PATH = "/home/szk_25/FederatedLLM"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)
if FEDERATEDLLM_PATH not in sys.path:
    sys.path.insert(0, FEDERATEDLLM_PATH)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_config_loading():
    """测试配置文件加载"""
    logger.info("Testing config loading...")
    
    config_path = "/home/szk_25/FedSA-LoRA-Dual/mmlu_evaluation_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return False
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info("Config loaded successfully")
        logger.info(f"Model path: {config['model']['checkpoint_path']}")
        logger.info(f"Test data path: {config['data']['test_data_path']}")
        logger.info(f"Output dir: {config['evaluation']['output_dir']}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    logger.info("Testing data loading...")
    
    data_path = "/home/szk_25/FederatedLLM/mmlu_test_1444.jsonl"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded successfully: {len(data)} samples")
        
        # 检查数据格式
        if len(data) > 0:
            sample = data[0]
            required_keys = ['instruction', 'input', 'output', 'class']
            for key in required_keys:
                if key not in sample:
                    logger.error(f"Missing required key: {key}")
                    return False
        
        logger.info("Data format validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return False

def test_model_path():
    """测试模型路径"""
    logger.info("Testing model path...")
    
    model_path = "/home/szk_25/FedSA-LoRA-Dual/exp/dual-lora_/home/szk_25/FederatedLLM/llama-7b@huggingface_llm_on_sst2@glue_lr0.0002_lstep10/sub_exp_20251012235324"
    
    if not os.path.exists(model_path):
        logger.warning(f"Model path does not exist: {model_path}")
        logger.info("This is expected if no training has been completed yet")
        return False
    
    # 检查是否有适配器权重文件
    adapter_weights_path = os.path.join(model_path, "adapter_model.bin")
    if os.path.exists(adapter_weights_path):
        logger.info("Adapter weights found")
    else:
        logger.warning("No adapter weights found")
    
    return True

def test_imports():
    """测试模块导入"""
    logger.info("Testing imports...")
    
    try:
        # 测试基础模块
        import torch
        import transformers
        import yaml
        import tqdm
        logger.info("Basic modules imported successfully")
        
        # 测试项目模块
        from fed_utils.evaluation import global_evaluation
        from utils.prompter import Prompter
        logger.info("Project modules imported successfully")
        
        # 测试双模块LoRA组件
        from code.dual_lora_model_builder import get_dual_lora_llm, DualLoRAModelBuilder
        from code.dual_lora_peft_adapter import DualLoraAdapterModel
        logger.info("Dual-LoRA modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def test_quick_evaluation():
    """测试快速评估功能"""
    logger.info("Testing quick evaluation...")
    
    # 检查快速评估脚本是否存在
    script_path = "/home/szk_25/FedSA-LoRA-Dual/quick_mmlu_eval.py"
    if not os.path.exists(script_path):
        logger.error(f"Quick evaluation script not found: {script_path}")
        return False
    
    logger.info("Quick evaluation script found")
    return True

def test_output_directory():
    """测试输出目录创建"""
    logger.info("Testing output directory creation...")
    
    output_dir = "/home/szk_25/FedSA-LoRA-Dual/mmlu_evaluation_results"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("Starting MMLU evaluation tests...")
    
    tests = [
        ("Config Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Model Path", test_model_path),
        ("Imports", test_imports),
        ("Quick Evaluation", test_quick_evaluation),
        ("Output Directory", test_output_directory)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info('='*50)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.warning(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # 打印测试摘要
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info('='*50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! MMLU evaluation is ready to use.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the issues above.")
    
    return results

def main():
    """主函数"""
    results = run_all_tests()
    
    # 保存测试结果
    results_file = "/home/szk_25/FedSA-LoRA-Dual/test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {results_file}")

if __name__ == '__main__':
    main()
