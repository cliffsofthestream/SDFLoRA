import os
import sys
import json
import pandas as pd
from collections import Counter

def debug_labels():
    print("=== 简化标签调试 ===")
    
    # 检查GLUE数据目录
    glue_dir = '/home/szk_25/FedSA-LoRA-Dual/GLUE'
    if not os.path.exists(glue_dir):
        print(f"GLUE目录不存在: {glue_dir}")
        return
    
    print(f"GLUE目录: {glue_dir}")
    
    # 查找MNLI数据
    mnli_dir = os.path.join(glue_dir, 'glue', 'mnli')
    if not os.path.exists(mnli_dir):
        print(f"MNLI目录不存在: {mnli_dir}")
        return
    
    print(f"MNLI目录: {mnli_dir}")
    
    # 查找版本目录
    version_dirs = [d for d in os.listdir(mnli_dir) if os.path.isdir(os.path.join(mnli_dir, d))]
    print(f"版本目录: {version_dirs}")
    
    if not version_dirs:
        print("没有找到版本目录")
        return
    
    # 使用第一个版本目录
    version_dir = os.path.join(mnli_dir, version_dirs[0])
    print(f"使用版本目录: {version_dir}")
    
    # 查找缓存目录
    cache_dirs = [d for d in os.listdir(version_dir) if os.path.isdir(os.path.join(version_dir, d))]
    print(f"缓存目录: {cache_dirs}")
    
    if not cache_dirs:
        print("没有找到缓存目录")
        return
    
    cache_dir = os.path.join(version_dir, cache_dirs[0])
    print(f"使用缓存目录: {cache_dir}")
    
    # 查找数据文件
    data_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            if file.endswith('.json') or file.endswith('.jsonl') or file.endswith('.parquet'):
                data_files.append(os.path.join(root, file))
    
    print(f"找到数据文件: {data_files}")
    
    if not data_files:
        print("没有找到数据文件")
        return
    
    # 分析标签
    labels = []
    for data_file in data_files:
        print(f"\n分析文件: {data_file}")
        try:
            if data_file.endswith('.parquet'):
                # Parquet格式
                df = pd.read_parquet(data_file)
                print(f"数据形状: {df.shape}")
                print(f"列名: {df.columns.tolist()}")
                if 'label' in df.columns:
                    labels.extend(df['label'].tolist())
                    print(f"标签列前5个值: {df['label'].head().tolist()}")
            elif data_file.endswith('.jsonl'):
                # JSONL格式
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if 'label' in data:
                                labels.append(data['label'])
            else:
                # JSON格式
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'label' in item:
                                labels.append(item['label'])
                    elif isinstance(data, dict):
                        if 'label' in data:
                            labels.append(data['label'])
        except Exception as e:
            print(f"读取文件失败: {e}")
    
    print(f"\n总共找到 {len(labels)} 个标签")
    
    if labels:
        # 统计标签分布
        label_counts = Counter(labels)
        print(f"标签分布: {dict(label_counts)}")
        
        # 获取唯一标签
        unique_labels = list(set(labels))
        print(f"唯一标签: {unique_labels}")
        print(f"标签数量: {len(unique_labels)}")
        
        # 检查标签类型
        print(f"标签类型: {[type(label).__name__ for label in unique_labels]}")
        
        # 检查是否有字符串标签
        string_labels = [label for label in unique_labels if isinstance(label, str)]
        if string_labels:
            print(f"字符串标签: {string_labels}")
        
        # 检查是否有数字标签
        numeric_labels = [label for label in unique_labels if isinstance(label, (int, float))]
        if numeric_labels:
            print(f"数字标签: {numeric_labels}")
    
    # 检查配置文件
    config_file = os.path.join(cache_dir, 'dataset_info.json')
    if os.path.exists(config_file):
        print(f"\n配置文件: {config_file}")
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"配置内容: {json.dumps(config, indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"读取配置文件失败: {e}")

if __name__ == "__main__":
    debug_labels()
