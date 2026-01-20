#!/usr/bin/env python3

import os
import sys
import subprocess
import requests
from pathlib import Path

def setup_tsinghua_mirror():
    """设置清华镜像源"""
    print("设置清华镜像源...")
    
    # 设置环境变量使用清华镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/szk_25/FedSA-LoRA/data/glue/'
    
    # 设置datasets缓存目录
    cache_dir = "/home/szk_25/FedSA-LoRA/data/glue/"
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"缓存目录: {cache_dir}")
    print(f"Hugging Face镜像: {os.environ['HF_ENDPOINT']}")

def download_mnli_with_tsinghua():
    """使用清华镜像源下载MNLI数据集"""
    
    setup_tsinghua_mirror()
    
    print("开始使用清华镜像源下载MNLI数据集...")
    
    try:
        # 导入datasets库
        from datasets import load_dataset
        
        print("正在从清华镜像源下载GLUE MNLI数据集...")
        
        # 使用清华镜像源下载
        dataset = load_dataset(
            "glue", 
            "mnli", 
            cache_dir="/home/szk_25/FedSA-LoRA/data/glue/",
            download_mode="reuse_dataset_if_exists"
        )
        
        print("✅ 数据集下载成功！")
        print(f"训练集大小: {len(dataset['train'])}")
        print(f"验证集(matched)大小: {len(dataset['validation_matched'])}")
        print(f"验证集(mismatched)大小: {len(dataset['validation_mismatched'])}")
        
        # 检查文件是否真的下载了
        cache_path = Path("/home/szk_25/FedSA-LoRA/data/glue/")
        if cache_path.exists():
            print(f"缓存文件位置: {cache_path}")
            for item in cache_path.rglob("*"):
                if item.is_file():
                    print(f"  - {item.name} ({item.stat().st_size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("尝试其他方法...")
        
        # 尝试使用huggingface-hub
        try:
            from huggingface_hub import hf_hub_download
            print("使用huggingface-hub下载...")
            
            # 下载GLUE数据集
            hf_hub_download(
                repo_id="glue",
                filename="mnli/train.json",
                cache_dir="/home/szk_25/FedSA-LoRA/data/glue/",
                endpoint="https://hf-mirror.com"
            )
            
            print("✅ 使用huggingface-hub下载成功！")
            return True
            
        except Exception as e2:
            print(f"❌ huggingface-hub方法也失败: {e2}")
            return False

def test_tsinghua_connection():
    """测试清华镜像源连接"""
    try:
        print("测试清华镜像源连接...")
        response = requests.get("https://hf-mirror.com", timeout=10)
        if response.status_code == 200:
            print("✅ 清华镜像源连接正常")
            return True
        else:
            print(f"❌ 镜像源响应异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到清华镜像源: {e}")
        return False

if __name__ == "__main__":
    print("=== 使用清华镜像源下载MNLI数据集 ===")
    
    # 测试镜像源连接
    if not test_tsinghua_connection():
        print("请检查网络连接或尝试其他镜像源")
        sys.exit(1)
    
    # 下载数据集
    success = download_mnli_with_tsinghua()
    
    if success:
        print("🎉 数据集下载完成！现在可以运行训练了")
    else:
        print("💥 数据集下载失败，请检查网络连接")
        print("可以尝试:")
        print("1. 检查网络连接")
        print("2. 使用VPN")
        print("3. 手动下载数据集文件")
