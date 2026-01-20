

import yaml
import sys
import os

def quick_verify():
    config_path = '/home/szk_25/FedSA-LoRA-Dual/dual_lora_config.yaml'
    
    print("=" * 60)
    print("快速验证配置文件")
    print("=" * 60)
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # 检查关键配置
    issues = []
    
    # 1. 检查method
    method = cfg.get('federate', {}).get('method', '')
    print(f"\n1. Federate method: {method}")
    if method != 'dual-lora':
        issues.append(f"❌ Method应该是'dual-lora'，当前是'{method}'")
    else:
        print("   ✅ Method正确")
    
    # 2. 检查num_labels
    num_labels = cfg.get('data', {}).get('num_labels')
    print(f"\n2. Data num_labels: {num_labels}")
    if num_labels is None:
        issues.append("❌ num_labels未设置")
    elif num_labels != 3:
        issues.append(f"⚠️  num_labels={num_labels}，MNLI应该是3")
    else:
        print("   ✅ num_labels正确设置为3")
    
    # 3. 检查聚合器配置
    aggregator = cfg.get('aggregator', {})
    robust_rule = aggregator.get('robust_rule', '')
    print(f"\n3. Aggregator robust_rule: {robust_rule}")
    if 'dual_lora' not in robust_rule.lower():
        issues.append(f"⚠️  robust_rule可能不匹配: {robust_rule}")
    else:
        print("   ✅ robust_rule包含dual_lora")
    
    # 总结
    print("\n" + "=" * 60)
    if issues:
        print("发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 配置检查通过")
    print("=" * 60)
    
    return len(issues) == 0

if __name__ == "__main__":
    quick_verify()

