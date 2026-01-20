#!/usr/bin/env python3
"""

setup:
python run_mmlu_experiments.py --config mmlu_evaluation_config.yaml
python run_mmlu_experiments.py --config mmlu_evaluation_config.yaml --variants dual_fusion,global_only
"""

import os
import sys
import yaml
import argparse
import logging
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MMLUExperimentRunner:
    """MMLU实验运行器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.results = {}
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run_single_experiment(self, variant: str, model_path: str) -> Dict[str, Any]:
        """运行单个实验"""
        logger.info(f"Running experiment: {variant}")
        
        # 创建变体特定的输出目录
        variant_output_dir = os.path.join(
            self.config['evaluation']['output_dir'],
            f"variant_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 构建命令
        cmd = [
            'python', 'mmlu_evaluator.py',
            '--config', self.config_path,
            '--model_path', model_path,
            '--output_dir', variant_output_dir,
            '--variant', variant
        ]
        
        # 运行实验
        try:
            logger.info(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
            
            if result.returncode == 0:
                logger.info(f"Experiment {variant} completed successfully")
                
                # 读取结果
                results_file = os.path.join(variant_output_dir, 'evaluation_summary.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        experiment_results = json.load(f)
                    experiment_results['variant'] = variant
                    experiment_results['model_path'] = model_path
                    experiment_results['status'] = 'success'
                    return experiment_results
                else:
                    logger.warning(f"No results file found for {variant}")
                    return {'variant': variant, 'status': 'no_results'}
            else:
                logger.error(f"Experiment {variant} failed: {result.stderr}")
                return {'variant': variant, 'status': 'failed', 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            logger.error(f"Experiment {variant} timed out")
            return {'variant': variant, 'status': 'timeout'}
        except Exception as e:
            logger.error(f"Experiment {variant} failed with exception: {e}")
            return {'variant': variant, 'status': 'error', 'error': str(e)}
    
    def run_all_experiments(self, variants: List[str] = None) -> Dict[str, Any]:
        """运行所有实验"""
        if variants is None:
            variants = list(self.config['experiment']['variants'].keys())
        
        logger.info(f"Running experiments for variants: {variants}")
        
        # 获取模型路径
        model_path = self.config['model']['checkpoint_path']
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            return {}
        
        # 运行每个变体
        for variant in variants:
            logger.info(f"Starting experiment: {variant}")
            result = self.run_single_experiment(variant, model_path)
            self.results[variant] = result
        
        # 保存汇总结果
        self.save_summary_results()
        
        return self.results
    
    def save_summary_results(self):
        """保存汇总结果"""
        summary_file = os.path.join(
            self.config['evaluation']['output_dir'],
            f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        summary = {
            'experiment_name': self.config['experiment']['name'],
            'description': self.config['experiment']['description'],
            'timestamp': datetime.now().isoformat(),
            'config_file': self.config_path,
            'model_path': self.config['model']['checkpoint_path'],
            'results': self.results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary results saved to {summary_file}")
        
        # 打印结果摘要
        self.print_summary()
    
    def print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*80)
        print("MMLU EXPERIMENT SUMMARY")
        print("="*80)
        
        for variant, result in self.results.items():
            print(f"\nVariant: {variant}")
            print("-" * 40)
            if result['status'] == 'success':
                print(f"Status: SUCCESS")
                print(f"Overall Accuracy: {result.get('overall_accuracy', 'N/A'):.4f}")
                print(f"STEM Accuracy: {result.get('stem_accuracy', 'N/A'):.4f}")
                print(f"Humanities Accuracy: {result.get('humanities_accuracy', 'N/A'):.4f}")
                print(f"Social Sciences Accuracy: {result.get('social_sciences_accuracy', 'N/A'):.4f}")
                print(f"Total Samples: {result.get('total_samples', 'N/A')}")
            else:
                print(f"Status: {result['status'].upper()}")
                if 'error' in result:
                    print(f"Error: {result['error']}")
        
        print("="*80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run MMLU Experiments')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--variants', type=str, help='Comma-separated list of variants to run')
    parser.add_argument('--model_path', type=str, help='Override model path')
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = MMLUExperimentRunner(args.config)
    
    # 覆盖模型路径
    if args.model_path:
        runner.config['model']['checkpoint_path'] = args.model_path
    
    # 确定要运行的变体
    variants = None
    if args.variants:
        variants = args.variants.split(',')
    
    # 运行实验
    results = runner.run_all_experiments(variants)
    
    logger.info("All experiments completed")

if __name__ == '__main__':
    main()
