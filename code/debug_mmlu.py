#!/usr/bin/env python3
"""
调试MMLU评估脚本
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_inference():
    """测试简单的模型推理"""
    logger.info("Testing simple model inference...")
    
    try:
        # 加载模型和分词器
        model_path = "/home/szk_25/FederatedLLM/llama-7b"
        
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 测试不同的提示格式
        test_prompts = [
            "### Instruction:\nWhat is the capital of France?\n\n### Input:\n\n### Response:\nThe answer is: ",
            "What is the capital of France? Answer:",
            "Question: What is the capital of France?\nAnswer:",
            "The capital of France is"
        ]
        
        for i, test_prompt in enumerate(test_prompts):
            logger.info(f"\n--- Testing prompt format {i+1} ---")
            logger.info(f"Prompt: {test_prompt}")
        
        logger.info(f"Test prompt: {test_prompt}")
        
        # 编码输入
        inputs = tokenizer(test_prompt, return_tensors="pt")
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        
        # 将输入移动到正确的设备
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        logger.info(f"Input device: {input_ids.device}, Model device: {device}")
        
        # 生成配置 - 使用更保守的设置
        generation_config = GenerationConfig(
            do_sample=False,  # 使用贪婪解码避免概率问题
            max_new_tokens=32,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 生成回答
        logger.info("Generating response...")
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 解码回答
        generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
        logger.info(f"Generated output: {generation_output_decoded}")
        
        # 提取回答部分
        ans = generation_output_decoded.split("### Response:")[-1].strip()
        logger.info(f"Extracted answer: {ans}")
        
        logger.info("Simple inference test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in simple inference test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    logger.info("Testing data loading...")
    
    try:
        data_path = "/home/szk_25/FederatedLLM/mmlu_test_1444.jsonl"
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                all_data = json.loads(content)
            else:
                all_data = []
                for line in content.split('\n'):
                    if line.strip():
                        all_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(all_data)} samples")
        
        # 检查第一个样本
        if all_data:
            sample = all_data[0]
            logger.info(f"First sample keys: {list(sample.keys())}")
            logger.info(f"First sample class: {sample.get('class')}")
            logger.info(f"First sample instruction: {sample.get('instruction', '')[:100]}...")
            logger.info(f"First sample input: {sample.get('input', '')[:100]}...")
            logger.info(f"First sample output: {sample.get('output', '')[:100]}...")
            
            # 测试答案解析
            target = sample["output"]
            logger.info(f"Target output: {target}")
            
            if "The answer is: " in target:
                tgt_ans_idx = target.replace('The answer is: ', '').split('. ')[0]
                tgt_ans = target.replace('The answer is: ', '').split('. ')[1]
                logger.info(f"Target answer index: {tgt_ans_idx}")
                logger.info(f"Target answer: {tgt_ans}")
            else:
                logger.warning("Target output doesn't contain 'The answer is: '")
        
        logger.info("Data loading test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in data loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_answer_matching():
    """测试答案匹配逻辑"""
    logger.info("Testing answer matching logic...")
    
    try:
        # 模拟一些测试案例
        test_cases = [
            {
                "target": "The answer is: A. Paris",
                "generated": "The answer is: A. Paris",
                "expected": True
            },
            {
                "target": "The answer is: B. London", 
                "generated": "The answer is: A. Paris",
                "expected": False
            },
            {
                "target": "The answer is: A. Paris",
                "generated": "A. Paris",
                "expected": True
            },
            {
                "target": "The answer is: A. Paris",
                "generated": "Paris",
                "expected": True
            }
        ]
        
        for i, case in enumerate(test_cases):
            target = case["target"]
            generated = case["generated"]
            expected = case["expected"]
            
            # 解析目标答案
            if "The answer is: " in target:
                tgt_ans_idx = target.replace('The answer is: ', '').split('. ')[0]
                tgt_ans = target.replace('The answer is: ', '').split('. ')[1]
            else:
                tgt_ans_idx = ""
                tgt_ans = target
            
            # 检查匹配
            is_correct = tgt_ans_idx + '.' in generated or tgt_ans in generated
            
            logger.info(f"Test case {i+1}:")
            logger.info(f"  Target: {target}")
            logger.info(f"  Generated: {generated}")
            logger.info(f"  Expected: {expected}, Got: {is_correct}")
            logger.info(f"  Match: {'✓' if is_correct == expected else '✗'}")
            logger.info("-" * 50)
        
        logger.info("Answer matching test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in answer matching test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting MMLU debug tests...")
    
    # 测试数据加载
    data_ok = test_data_loading()
    
    # 测试答案匹配
    matching_ok = test_answer_matching()
    
    # 测试简单推理
    inference_ok = test_simple_inference()
    
    logger.info("=" * 50)
    logger.info("Debug test results:")
    logger.info(f"Data loading: {'✓' if data_ok else '✗'}")
    logger.info(f"Answer matching: {'✓' if matching_ok else '✗'}")
    logger.info(f"Simple inference: {'✓' if inference_ok else '✗'}")
    
    if all([data_ok, matching_ok, inference_ok]):
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed!")
