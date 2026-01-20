import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_model_output_quality():
    """测试模型输出质量"""
    logger.info("Testing model output quality...")
    
    try:
        # 加载模型和分词器
        model_path = "/home/user/FederatedLLM/llama-7b"
        
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置pad_token 并统一到模型配置
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, "pad_token_id", None) is None or model.config.pad_token_id == -1:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None or model.config.eos_token_id == -1:
            model.config.eos_token_id = tokenizer.eos_token_id
        
        # 测试不同的提示格式
        test_cases = [
            {
                "name": "Original format",
                "prompt": "### Instruction:\nWhat is the capital of France?\n\n### Input:\n\n### Response:\nThe answer is: "
            },
            {
                "name": "Simple format",
                "prompt": "What is the capital of France? Answer:"
            },
            {
                "name": "Question format", 
                "prompt": "Question: What is the capital of France?\nAnswer:"
            },
            {
                "name": "Direct format",
                "prompt": "The capital of France is"
            }
        ]
        
        device = next(model.parameters()).device
        logger.info(f"Model device: {device}")
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i+1}: {test_case['name']}")
            logger.info(f"Prompt: {test_case['prompt']}")
            
            # 编码输入
            inputs = tokenizer(test_case['prompt'], return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = torch.ones_like(input_ids)
            
            # 生成配置
            generation_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 生成回答
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    max_new_tokens=64,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码回答
            full_output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            logger.info(f"Full output: {full_output}")
            
            # 提取新生成的部分
            new_tokens = generation_output.sequences[0][input_ids.shape[1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"New generated text: '{new_text}'")
            
            # 检查是否包含unk标记
            unk_count = new_text.count('<unk>')
            if unk_count > 0:
                logger.warning(f"⚠️  Generated text contains {unk_count} <unk> tokens")
            else:
                logger.info("✅ No <unk> tokens in generated text")
        
        logger.info(f"\n{'='*60}")
        logger.info("Model output quality test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in model output quality test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting model output quality test...")
    success = test_model_output_quality()
    
    if success:
        logger.info("✅ All tests completed successfully!")
    else:
        logger.error("❌ Some tests failed!")
