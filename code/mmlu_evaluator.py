#!/usr/bin/env python3
"""
MMLU评估脚本 - 支持双模块LoRA模型
基于FederatedLLM的评估框架，扩展支持双模块LoRA架构

使用方法:
python mmlu_evaluator.py --config mmlu_evaluation_config.yaml
python mmlu_evaluator.py --config mmlu_evaluation_config.yaml --variant dual_fusion
python mmlu_evaluator.py --model_path /path/to/model --tasks mmlu_abstract_algebra,mmlu_anatomy
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig,
    BitsAndBytesConfig
)

# 添加项目路径
CURRENT_PROJECT_PATH = "/home/szk_25/FedSA-LoRA-Dual"
FEDERATEDLLM_PATH = "/home/szk_25/FederatedLLM"
if CURRENT_PROJECT_PATH not in sys.path:
    sys.path.insert(0, CURRENT_PROJECT_PATH)
if FEDERATEDLLM_PATH not in sys.path:
    sys.path.insert(0, FEDERATEDLLM_PATH)

# 导入双模块LoRA组件
from code.dual_lora_model_builder import get_dual_lora_llm, DualLoRAModelBuilder
from code.dual_lora_peft_adapter import DualLoraAdapterModel
from fed_utils.evaluation import global_evaluation
from utils.prompter import Prompter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MMLUConfig:
    """MMLU评估配置"""
    base_model_path: str
    adapter_path: Optional[str] = None
    test_data_path: str = ""
    output_dir: str = ""
    tasks: List[str] = None
    batch_size: int = 8
    max_new_tokens: int = 32
    temperature: float = 0.2
    top_p: float = 0.6
    top_k: int = 30
    num_beams: int = 1
    use_dual_lora: bool = True
    fusion_method: str = 'weighted_sum'
    save_detailed_results: bool = True
    verbose: bool = False
    
    @property
    def model_path(self):
        """兼容性属性，返回基础模型路径"""
        return self.base_model_path

class MMLUDataset(Dataset):
    """MMLU数据集类"""
    
    def __init__(self, data_path: str, tasks: Optional[List[str]] = None, 
                 max_samples: Optional[int] = None, sample_ratio: float = 1.0):
        self.data_path = data_path
        self.tasks = tasks
        self.data = []
        self.load_data(max_samples, sample_ratio)
    
    def load_data(self, max_samples: Optional[int] = None, sample_ratio: float = 1.0):
        """加载MMLU数据"""
        logger.info(f"Loading MMLU data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                # JSON数组格式
                all_data = json.loads(content)
            else:
                # JSONL格式
                all_data = []
                for line in content.split('\n'):
                    if line.strip():
                        all_data.append(json.loads(line))
        
        # 过滤任务
        if self.tasks:
            filtered_data = []
            for item in all_data:
                if item.get('class') in self.tasks:
                    filtered_data.append(item)
            all_data = filtered_data
        
        # 采样
        if sample_ratio < 1.0:
            sample_size = int(len(all_data) * sample_ratio)
            all_data = all_data[:sample_size]
        
        # 限制样本数量
        if max_samples:
            all_data = all_data[:max_samples]
        
        self.data = all_data
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class MMLUEvaluator:
    """MMLU评估器"""
    
    def __init__(self, config: MMLUConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.prompter = None
        self.results = {}
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(config.output_dir, 'evaluation.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def load_model(self):
        """加载模型和分词器"""
        logger.info(f"Loading base model from {self.config.base_model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 在加载前，若配置的是子实验根目录，则尝试自动链接/复制最新一轮权重到根目录
        if self.config.adapter_path:
            try:
                root_dir = Path(self.config.adapter_path)
                if root_dir.exists() and root_dir.is_dir():
                    root_adapter = root_dir / "adapter_model.bin"
                    # 已存在则不处理
                    if not root_adapter.exists():
                        # 收集形如 outdir/<epoch>/adapter_model.bin 的候选
                        candidates = []
                        for child in root_dir.iterdir():
                            if child.is_dir() and child.name.isdigit():
                                adapter_file = child / "adapter_model.bin"
                                if adapter_file.exists():
                                    try:
                                        epoch_num = int(child.name)
                                        candidates.append((epoch_num, adapter_file))
                                    except ValueError:
                                        pass
                        if candidates:
                            candidates.sort(key=lambda x: x[0])
                            latest_epoch, latest_adapter = candidates[-1]
                            try:
                                # 优先尝试软链接，失败则复制
                                os.symlink(str(latest_adapter), str(root_adapter))
                                logger.info(f"Symlinked latest adapter (epoch {latest_epoch}) to root: {root_adapter}")
                            except Exception:
                                try:
                                    import shutil
                                    shutil.copy2(str(latest_adapter), str(root_adapter))
                                    logger.info(f"Copied latest adapter (epoch {latest_epoch}) to root: {root_adapter}")
                                except Exception as copy_err:
                                    logger.warning(f"Failed to place adapter_model.bin at root: {copy_err}")
            except Exception as e:
                logger.warning(f"Auto-link latest adapter failed: {e}")

        # 检查是否有适配器路径
        if self.config.adapter_path and os.path.exists(self.config.adapter_path):
            logger.info(f"Loading adapter from {self.config.adapter_path}")
            if self.config.use_dual_lora:
                # 加载双模块LoRA模型
                self.model = self._load_dual_lora_model()
            else:
                # 加载标准适配器模型
                self.model = self._load_adapter_model()
        else:
            logger.info("No adapter path or adapter not found, loading base model only")
            # 加载标准模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # 加载提示器
        try:
            self.prompter = Prompter("alpaca")
        except ValueError as e:
            logger.warning(f"Failed to load prompter: {e}")
            logger.info("Using default prompter")
            self.prompter = None
        
        logger.info("Model loaded successfully")
    
    def _append_jsonl(self, file_path: str, record: Dict[str, Any]):
        """将记录以一行JSON追加到JSONL文件。"""
        # 目录保证存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _atomic_write_json(self, file_path: str, content: Dict[str, Any]):
        """以原子方式写入JSON（先写临时文件，再替换）。"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp_path = file_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, file_path)

    def _load_adapter_model(self):
        """加载标准适配器模型"""
        try:
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载适配器权重
            adapter_weights_path = os.path.join(self.config.adapter_path, "adapter_model.bin")
            if os.path.exists(adapter_weights_path):
                logger.info(f"Loading adapter weights from {adapter_weights_path}")
                adapter_weights = torch.load(adapter_weights_path, map_location='cpu')
                model.load_state_dict(adapter_weights, strict=False)
            else:
                logger.warning(f"Adapter weights not found at {adapter_weights_path}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to load adapter model: {e}")
            logger.info("Falling back to base model")
            return AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    
    def _load_dual_lora_model(self):
        """加载双模块LoRA模型"""
        try:
            # 加载基础模型
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 创建双模块LoRA适配器
            dual_lora_config = {
                'adapter_package': 'dual_lora',
                'adapter_method': 'dual_lora',
                'use_dual_lora': True,
                'global_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'local_r': 4,
                'fusion_method': self.config.fusion_method,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            }
            
            # 应用双模块LoRA适配器
            model = DualLoraAdapterModel(base_model, dual_lora_config)
            
            # 加载适配器权重（如果存在）
            adapter_weights_path = os.path.join(self.config.adapter_path, "adapter_model.bin")
            if os.path.exists(adapter_weights_path):
                logger.info(f"Loading adapter weights from {adapter_weights_path}")
                adapter_weights = torch.load(adapter_weights_path, map_location='cpu')
                model.load_state_dict(adapter_weights, strict=False)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load dual LoRA model: {e}")
            logger.info("Falling back to standard model loading")
            
            # 回退到标准模型加载
            return AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    
    def evaluate_single_task(self, task_data: List[Dict]) -> Dict[str, Any]:
        """评估单个任务"""
        logger.info(f"Evaluating task with {len(task_data)} samples")
        
        # 按类别分组
        class_data = {}
        for item in task_data:
            class_name = item.get('class', 'unknown')
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append(item)
        
        # 评估每个类别
        class_results = {}
        total_correct = 0
        total_samples = 0
        
        for class_name, items in class_data.items():
            correct, total = self._evaluate_class(class_name, items)
            class_results[class_name] = {
                'correct': correct,
                'total': total,
                'accuracy': correct / total if total > 0 else 0.0
            }
            total_correct += correct
            total_samples += total
        
        # 计算总体准确率
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'class_results': class_results
        }
    
    def _evaluate_class(self, class_name: str, items: List[Dict]) -> Tuple[int, int]:
        """评估单个类别"""
        correct = 0
        total = len(items)
        
        # 生成配置
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            num_beams=1,
            max_new_tokens=min(8, max(1, self.config.max_new_tokens)),
            early_stopping=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        for item in tqdm(items, desc=f"Evaluating {class_name}"):
            try:
                # 获取目标答案
                target = item["output"]
                tgt_ans_idx = target.replace('The answer is: ', '').split('. ')[0]
                tgt_ans = target.replace('The answer is: ', '').split('. ')[1]
                
                # 生成提示
                if self.prompter:
                    test_prompt = self.prompter.generate_prompt(
                        item["instruction"],
                        item["input"],
                        'The answer is: ',
                    )
                else:
                    # 使用简单的提示格式
                    test_prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\nThe answer is: "
                
                # 生成回答
                inputs = self.tokenizer(test_prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                with torch.no_grad():
                    generation_output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                        max_new_tokens=generation_config.max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码回答
                generation_output_decoded = self.tokenizer.decode(generation_output.sequences[0])
                if self.prompter:
                    split = self.prompter.template["response_split"]
                    ans = generation_output_decoded.split(split)[-1].strip()
                else:
                    # 使用简单的分割方式
                    ans = generation_output_decoded.split("### Response:")[-1].strip()
                
                # 解析答案，优先提取 A-D 字母
                ans_upper = ans.strip().upper()
                extracted = None
                # 常见格式："THE ANSWER IS: X"、"ANSWER: X"、行首/句首的选项字母
                for token in ["THE ANSWER IS:", "ANSWER:", "THE CORRECT ANSWER IS:"]:
                    if token in ans_upper:
                        after = ans_upper.split(token, 1)[1].strip()
                        if after:
                            extracted = after[0]
                            break
                if extracted is None and len(ans_upper) > 0:
                    extracted = ans_upper[0]
                is_correct = False
                if extracted in ["A", "B", "C", "D"]:
                    is_correct = (extracted == tgt_ans_idx.strip().upper())
                else:
                    # 回退到包含关系的宽松匹配
                    is_correct = (tgt_ans_idx + '.' in ans) or (tgt_ans in ans)
                if is_correct:
                    correct += 1
                
                if self.config.verbose:
                    logger.info(f"Question: {item['input'][:100]}...")
                    logger.info(f"Target: {tgt_ans}")
                    logger.info(f"Generated: {ans}")
                    logger.info(f"Parsed option: {extracted}")
                    logger.info(f"Correct: {is_correct}")
                    logger.info("-" * 50)
                    
            except Exception as e:
                logger.error(f"Error evaluating sample: {e}")
                continue
        
        return correct, total
    
    def evaluate_all_tasks(self) -> Dict[str, Any]:
        """评估所有任务"""
        logger.info("Starting MMLU evaluation")
        
        # 加载数据
        dataset = MMLUDataset(
            self.config.test_data_path,
            tasks=self.config.tasks
        )
        
        # 按任务分组
        task_data = {}
        for item in dataset.data:
            class_name = item.get('class', 'unknown')
            if class_name not in task_data:
                task_data[class_name] = []
            task_data[class_name].append(item)
        
        # 任务类别列表（用于分类准确率计算）
        stem_tasks = ['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 
                     'college_chemistry', 'college_computer_science', 'college_mathematics',
                     'college_medicine', 'college_physics', 'computer_security',
                     'conceptual_physics', 'electrical_engineering', 'elementary_mathematics',
                     'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
                     'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
                     'machine_learning', 'medical_genetics', 'virology']

        humanities_tasks = ['high_school_european_history', 'high_school_us_history',
                           'high_school_world_history', 'philosophy', 'prehistory', 'world_religions']

        social_sciences_tasks = ['business_ethics', 'high_school_geography',
                                'high_school_government_and_politics', 'high_school_macroeconomics',
                                'high_school_microeconomics', 'high_school_psychology',
                                'econometrics', 'management', 'marketing', 'professional_accounting',
                                'professional_psychology', 'public_relations', 'sociology', 'us_foreign_policy']

        # 评估每个任务，增量写盘
        task_results = {}
        for task_name, items in task_data.items():
            logger.info(f"Evaluating task: {task_name}")
            single_result = self.evaluate_single_task(items)
            task_results[task_name] = single_result

            # 追加写入单任务结果到 JSONL
            per_task_path = os.path.join(self.config.output_dir, 'per_task_results.jsonl')
            self._append_jsonl(per_task_path, {
                'task_name': task_name,
                'result': single_result,
                'timestamp': datetime.now().isoformat()
            })

            # 计算到目前为止的部分统计并原子写入 partial JSON
            total_correct_partial = sum(r['total_correct'] for r in task_results.values())
            total_samples_partial = sum(r['total_samples'] for r in task_results.values())
            overall_accuracy_partial = (
                total_correct_partial / total_samples_partial if total_samples_partial > 0 else 0.0
            )

            stem_accuracy_partial = self._calculate_category_accuracy(task_results, stem_tasks)
            humanities_accuracy_partial = self._calculate_category_accuracy(task_results, humanities_tasks)
            social_sciences_accuracy_partial = self._calculate_category_accuracy(task_results, social_sciences_tasks)

            partial_results = {
                'overall_accuracy': overall_accuracy_partial,
                'total_correct': total_correct_partial,
                'total_samples': total_samples_partial,
                'stem_accuracy': stem_accuracy_partial,
                'humanities_accuracy': humanities_accuracy_partial,
                'social_sciences_accuracy': social_sciences_accuracy_partial,
                'task_results': task_results,
                'evaluation_time': datetime.now().isoformat(),
                'config': {
                    'model_path': self.config.model_path,
                    'tasks': self.config.tasks,
                    'fusion_method': self.config.fusion_method,
                    'use_dual_lora': self.config.use_dual_lora
                }
            }

            # 更新内存中的当前结果，便于需要时打印
            self.results = partial_results

            # 原子写入部分结果和部分汇总
            partial_results_path = os.path.join(self.config.output_dir, 'evaluation_results_partial.json')
            self._atomic_write_json(partial_results_path, partial_results)

            partial_summary = {
                'overall_accuracy': overall_accuracy_partial,
                'stem_accuracy': stem_accuracy_partial,
                'humanities_accuracy': humanities_accuracy_partial,
                'social_sciences_accuracy': social_sciences_accuracy_partial,
                'total_samples': total_samples_partial,
                'evaluation_time': partial_results['evaluation_time']
            }
            partial_summary_path = os.path.join(self.config.output_dir, 'evaluation_summary_partial.json')
            self._atomic_write_json(partial_summary_path, partial_summary)

        # 完整统计（全部任务结束）
        total_correct = sum(result['total_correct'] for result in task_results.values())
        total_samples = sum(result['total_samples'] for result in task_results.values())
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        stem_accuracy = self._calculate_category_accuracy(task_results, stem_tasks)
        humanities_accuracy = self._calculate_category_accuracy(task_results, humanities_tasks)
        social_sciences_accuracy = self._calculate_category_accuracy(task_results, social_sciences_tasks)

        self.results = {
            'overall_accuracy': overall_accuracy,
            'total_correct': total_correct,
            'total_samples': total_samples,
            'stem_accuracy': stem_accuracy,
            'humanities_accuracy': humanities_accuracy,
            'social_sciences_accuracy': social_sciences_accuracy,
            'task_results': task_results,
            'evaluation_time': datetime.now().isoformat(),
            'config': {
                'model_path': self.config.model_path,
                'tasks': self.config.tasks,
                'fusion_method': self.config.fusion_method,
                'use_dual_lora': self.config.use_dual_lora
            }
        }

        return self.results
    
    def _calculate_category_accuracy(self, task_results: Dict, category_tasks: List[str]) -> float:
        """计算类别准确率"""
        total_correct = 0
        total_samples = 0
        
        for task_name, result in task_results.items():
            if task_name in category_tasks:
                total_correct += result['total_correct']
                total_samples += result['total_samples']
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def save_results(self):
        """保存评估结果"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # 保存详细结果
        results_file = os.path.join(self.config.output_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 保存简化结果
        summary_file = os.path.join(self.config.output_dir, 'evaluation_summary.json')
        summary = {
            'overall_accuracy': self.results['overall_accuracy'],
            'stem_accuracy': self.results['stem_accuracy'],
            'humanities_accuracy': self.results['humanities_accuracy'],
            'social_sciences_accuracy': self.results['social_sciences_accuracy'],
            'total_samples': self.results['total_samples'],
            'evaluation_time': self.results['evaluation_time']
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印结果
        self.print_results()
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def print_results(self):
        """打印评估结果"""
        if not self.results:
            logger.warning("No results to print")
            return
        
        print("\n" + "="*60)
        print("MMLU EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Accuracy: {self.results['overall_accuracy']:.4f}")
        print(f"STEM Accuracy: {self.results['stem_accuracy']:.4f}")
        print(f"Humanities Accuracy: {self.results['humanities_accuracy']:.4f}")
        print(f"Social Sciences Accuracy: {self.results['social_sciences_accuracy']:.4f}")
        print(f"Total Samples: {self.results['total_samples']}")
        print(f"Evaluation Time: {self.results['evaluation_time']}")
        print("="*60)
        
        # 打印各任务结果
        print("\nPer-Task Results:")
        print("-" * 40)
        for task_name, result in self.results['task_results'].items():
            print(f"{task_name:30s}: {result['overall_accuracy']:.4f} ({result['total_correct']}/{result['total_samples']})")

def load_config(config_path: str) -> MMLUConfig:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 提取评估配置
    eval_config = config_dict.get('evaluation', {})
    data_config = config_dict.get('data', {})
    model_config = config_dict.get('model', {})
    
    # 获取任务列表
    tasks = []
    if 'tasks' in eval_config:
        for category, task_list in eval_config['tasks'].items():
            if isinstance(task_list, list):
                tasks.extend(task_list)
    
    # 创建配置对象
    config = MMLUConfig(
        base_model_path=model_config.get('base_model_path', model_config.get('type', '')),
        adapter_path=model_config.get('adapter_path'),
        test_data_path=data_config.get('test_data_path', ''),
        output_dir=eval_config.get('output_dir', './mmlu_results'),
        tasks=tasks,
        batch_size=eval_config.get('batch_size', 8),
        max_new_tokens=eval_config.get('max_new_tokens', 32),
        temperature=eval_config.get('temperature', 0.2),
        top_p=eval_config.get('top_p', 0.6),
        top_k=eval_config.get('top_k', 30),
        num_beams=eval_config.get('num_beams', 1),
        use_dual_lora=eval_config.get('use_dual_lora', True),
        fusion_method=eval_config.get('fusion_method', 'weighted_sum'),
        save_detailed_results=eval_config.get('save_detailed_results', True),
        verbose=eval_config.get('verbose', False)
    )
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MMLU Evaluation for Dual-LoRA Models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Override model path')
    parser.add_argument('--tasks', type=str, help='Comma-separated list of tasks to evaluate')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--variant', type=str, help='Evaluation variant')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置
    if args.model_path:
        # model_path 是只读属性（映射到 base_model_path），因此这里应设置 base_model_path
        config.base_model_path = args.model_path
    if args.tasks:
        config.tasks = args.tasks.split(',')
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.verbose:
        config.verbose = True
    
    # 根据变体调整配置
    if args.variant:
        if args.variant == 'global_only':
            config.use_dual_lora = False
        elif args.variant == 'local_only':
            config.fusion_method = 'local_only'
        elif args.variant == 'dual_fusion':
            config.fusion_method = 'weighted_sum'
        elif args.variant == 'dual_gating':
            config.fusion_method = 'gating'
    
    # 创建评估器
    evaluator = MMLUEvaluator(config)
    
    # 加载模型
    evaluator.load_model()
    
    # 运行评估
    start_time = time.time()
    results = evaluator.evaluate_all_tasks()
    end_time = time.time()
    
    # 保存结果
    evaluator.save_results()
    
    logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
