

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def debug_cuda_error():
    """调试CUDA错误"""
    logger.info("Debugging CUDA error...")
    
    try:
        # 加载模型和分词器
        model_path = "/home/szk_25/FederatedLLM/llama-7b"
        
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 检查tokenizer配置
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        logger.info(f"Tokenizer model max length: {tokenizer.model_max_length}")
        logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")
        logger.info(f"Tokenizer eos token: {tokenizer.eos_token}")
        logger.info(f"Tokenizer eos token id: {tokenizer.eos_token_id}")
        logger.info(f"Tokenizer unk token: {tokenizer.unk_token}")
        logger.info(f"Tokenizer unk token id: {tokenizer.unk_token_id}")
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 检查模型配置
        logger.info(f"Model config vocab size: {model.config.vocab_size}")
        logger.info(f"Model config pad token id: {model.config.pad_token_id}")
        logger.info(f"Model config eos token id: {model.config.eos_token_id}")
        
        # 检查词汇表大小是否匹配
        if tokenizer.vocab_size != model.config.vocab_size:
            logger.error(f"❌ Vocab size mismatch! Tokenizer: {tokenizer.vocab_size}, Model: {model.config.vocab_size}")
            return False
        else:
            logger.info("✅ Vocab sizes match")
        
        # 测试简单的tokenization
        test_text = "Hello world"
        logger.info(f"Testing tokenization with: '{test_text}'")
        
        inputs = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Input IDs: {inputs['input_ids']}")
        logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
        logger.info(f"Max token ID: {inputs['input_ids'].max().item()}")
        logger.info(f"Min token ID: {inputs['input_ids'].min().item()}")
        
        # 检查是否有超出词汇表的token
        if inputs['input_ids'].max().item() >= tokenizer.vocab_size:
            logger.error(f"❌ Token ID {inputs['input_ids'].max().item()} >= vocab_size {tokenizer.vocab_size}")
            return False
        
        # 测试解码
        decoded = tokenizer.decode(inputs['input_ids'][0])
        logger.info(f"Decoded text: '{decoded}'")
        
        # 测试更简单的生成
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")
        
        # 使用最简单的提示
        simple_prompt = "The capital of France is"
        logger.info(f"Testing with simple prompt: '{simple_prompt}'")
        
        inputs = tokenizer(simple_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # 检查输入token是否在有效范围内
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()
        logger.info(f"Input token range: {min_token_id} to {max_token_id}")
        
        if max_token_id >= tokenizer.vocab_size:
            logger.error(f"❌ Input contains token ID {max_token_id} >= vocab_size {tokenizer.vocab_size}")
            return False
        
        # 使用最保守的生成设置
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=5,  # 只生成5个token
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        
        logger.info("Attempting generation with conservative settings...")
        
        # 设置CUDA错误检查
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                max_new_tokens=5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码回答
        full_output = tokenizer.decode(generation_output.sequences[0])
        logger.info(f"Full output: {full_output}")
        
        # 提取新生成的部分
        new_tokens = generation_output.sequences[0][input_ids.shape[1]:]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.info(f"New generated text: '{new_text}'")
        
        logger.info("✅ Simple generation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in CUDA debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting CUDA error debug...")
    success = debug_cuda_error()
    
    if success:
        logger.info("✅ Debug completed successfully!")
    else:
        logger.error("❌ Debug failed!")
