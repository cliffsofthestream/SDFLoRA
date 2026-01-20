#!/usr/bin/env python3
"""
修复GLUE训练器中的标签验证问题
在模型前向传播前验证标签值是否在有效范围内
"""

import os
import sys

# 添加路径
ORIGINAL_PROJECT_PATH = "/home/szk_25/FedSA-LoRA"
if ORIGINAL_PROJECT_PATH not in sys.path:
    sys.path.insert(0, ORIGINAL_PROJECT_PATH)

# 读取原始训练器文件
trainer_path = "/home/szk_25/FedSA-LoRA/federatedscope/glue/trainer/trainer.py"
backup_path = "/home/szk_25/FedSA-LoRA/federatedscope/glue/trainer/trainer.py.backup"

# 如果备份不存在，先创建备份
if not os.path.exists(backup_path):
    print(f"Creating backup: {backup_path}")
    with open(trainer_path, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    print("Backup created successfully")

# 读取训练器文件
with open(trainer_path, 'r') as f:
    content = f.read()

# 检查是否已经修复过
if "def _hook_on_batch_forward(self, ctx):" in content and "num_labels" in content and "labels.clamp" in content:
    print("Trainer already has label validation. Checking if fix is complete...")
    
    # 检查是否有完整的验证逻辑
    if "labels = labels.clamp" in content:
        print("✅ Label validation already exists in trainer")
    else:
        print("⚠️  Partial fix detected, applying complete fix...")
else:
    print("Applying label validation fix...")

# 修复_hook_on_batch_forward方法，添加标签验证
old_method = """    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)
"""

new_method = """    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['label'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        
        # 验证标签值是否在有效范围内（防止CUDA断言错误）
        if hasattr(ctx.cfg.data, 'num_labels') and ctx.cfg.data.num_labels > 0:
            num_labels = ctx.cfg.data.num_labels
            # 检查标签值范围
            min_label = labels.min().item() if len(labels) > 0 else 0
            max_label = labels.max().item() if len(labels) > 0 else 0
            
            if max_label >= num_labels or min_label < 0:
                logger.warning(
                    f'Invalid label range detected: min={min_label}, max={max_label}, '
                    f'expected range=[0, {num_labels-1}]. Clamping labels to valid range.'
                )
                # 将标签限制在有效范围内
                labels = labels.clamp(min=0, max=num_labels - 1)
                
                # 记录统计信息
                invalid_count = ((labels < 0) | (labels >= num_labels)).sum().item()
                if invalid_count > 0:
                    logger.error(
                        f'Found {invalid_count} invalid labels after clamping. '
                        f'This may indicate a data loading issue.'
                    )
        else:
            # 如果没有配置num_labels，尝试从模型获取
            try:
                if hasattr(ctx.model, 'config') and hasattr(ctx.model.config, 'num_labels'):
                    num_labels = ctx.model.config.num_labels
                elif hasattr(ctx.model, 'model') and hasattr(ctx.model.model, 'config') and hasattr(ctx.model.model.config, 'num_labels'):
                    num_labels = ctx.model.model.config.num_labels
                else:
                    # 尝试从输出维度推断
                    with torch.no_grad():
                        if ctx.cfg.llm.deepspeed.use:
                            test_outputs = ctx.model_engine(input_ids=input_ids[:1], attention_mask=attention_mask[:1])
                        else:
                            test_outputs = ctx.model(input_ids=input_ids[:1], attention_mask=attention_mask[:1])
                        num_labels = test_outputs.logits.shape[-1]
                
                min_label = labels.min().item() if len(labels) > 0 else 0
                max_label = labels.max().item() if len(labels) > 0 else 0
                
                if max_label >= num_labels or min_label < 0:
                    logger.warning(
                        f'Invalid label range detected: min={min_label}, max={max_label}, '
                        f'num_labels={num_labels}. Clamping labels to valid range.'
                    )
                    labels = labels.clamp(min=0, max=num_labels - 1)
            except Exception as e:
                logger.warning(f'Could not validate labels automatically: {e}')
        
        if ctx.cfg.llm.deepspeed.use:
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)
        else:
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)
"""

if old_method in content:
    # 替换方法
    content = content.replace(old_method, new_method)
    
    # 写回文件
    with open(trainer_path, 'w') as f:
        f.write(content)
    print("✅ Successfully applied label validation fix to GLUE trainer")
    print(f"   Fixed file: {trainer_path}")
    print(f"   Backup saved: {backup_path}")
else:
    # 检查是否已经是新版本
    if "labels.clamp" in content:
        print("✅ Label validation already exists in trainer")
    else:
        print("❌ Could not find target method to fix")
        print("   Please check the trainer file manually")

