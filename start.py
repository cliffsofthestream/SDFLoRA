
import os
import sys
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """检查依赖是否安装"""
    logger.info("检查依赖...")
    
    required_packages = [
        "torch",
        "transformers", 
        "peft",
        "numpy",
        "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.info("请运行: pip install torch transformers peft numpy PyYAML")
        return False
    
    logger.info("所有依赖检查通过!")
    return True

def check_original_project():
    """检查原项目是否存在"""
    original_path = "/home/szk_25/FedSA-LoRA"
    
    if os.path.exists(original_path):
        logger.info(f"✓ 原项目路径存在: {original_path}")
        return True
    else:
        logger.warning(f"✗ 原项目路径不存在: {original_path}")
        logger.info("双模块LoRA可以独立运行，但某些功能可能受限")
        return False

def run_tests():
    """运行测试"""
    logger.info("运行双模块LoRA测试...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_dual_lora.py"
        ], capture_output=True, text=True, cwd="/home/szk_25/FedSA-LoRA-Dual")
        
        if result.returncode == 0:
            logger.info("✓ 所有测试通过!")
            return True
        else:
            logger.error(f"✗ 测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"运行测试时出错: {e}")
        return False

def run_examples():
    """运行示例"""
    logger.info("运行使用示例...")
    
    try:
        result = subprocess.run([
            sys.executable, "example_usage.py"
        ], capture_output=True, text=True, cwd="/home/szk_25/FedSA-LoRA-Dual")
        
        if result.returncode == 0:
            logger.info("✓ 示例运行成功!")
            return True
        else:
            logger.error(f"✗ 示例运行失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        return False

def interactive_demo():
    """交互式演示"""
    logger.info("启动交互式演示...")
    
    print("\n" + "="*60)
    print("双模块LoRA交互式演示")
    print("="*60)
    
    while True:
        print("\n请选择操作:")
        print("1. 运行基础测试")
        print("2. 运行使用示例") 
        print("3. 查看配置文件")
        print("4. 查看项目结构")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            run_tests()
        elif choice == "2":
            run_examples()
        elif choice == "3":
            show_configs()
        elif choice == "4":
            show_project_structure()
        elif choice == "5":
            logger.info("退出演示")
            break
        else:
            print("无效选择，请重新输入")

def show_configs():
    """显示配置文件"""
    logger.info("显示配置文件...")
    
    config_files = [
        "dual_lora_config.yaml",
        "dual_lora_hetero_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = f"/home/szk_25/FedSA-LoRA-Dual/{config_file}"
        if os.path.exists(config_path):
            print(f"\n--- {config_file} ---")
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 只显示前20行
                lines = content.split('\n')[:20]
                print('\n'.join(lines))
                if len(content.split('\n')) > 20:
                    print("... (更多内容请查看完整文件)")
        else:
            logger.warning(f"配置文件不存在: {config_file}")

def show_project_structure():
    """显示项目结构"""
    logger.info("显示项目结构...")
    
    project_path = "/home/szk_25/FedSA-LoRA-Dual"
    
    print(f"\n项目结构: {project_path}")
    print("-" * 50)
    
    for root, dirs, files in os.walk(project_path):
        level = root.replace(project_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                print(f"{subindent}{file}")

def main():
    """主函数"""
    logger.info("双模块LoRA快速开始")
    logger.info("="*50)
    
    # 检查环境
    if not check_dependencies():
        return False
    
    check_original_project()
    
    # 询问用户想要做什么
    print("\n欢迎使用双模块LoRA!")
    print("请选择启动模式:")
    print("1. 自动运行测试和示例")
    print("2. 交互式演示")
    print("3. 仅运行测试")
    print("4. 仅运行示例")
    
    choice = input("\n请输入选择 (1-4): ").strip()
    
    if choice == "1":
        logger.info("自动运行模式")
        success = True
        success &= run_tests()
        success &= run_examples()
        
        if success:
            logger.info("🎉 所有操作完成!")
        else:
            logger.error("❌ 某些操作失败")
        
        return success
        
    elif choice == "2":
        interactive_demo()
        return True
        
    elif choice == "3":
        return run_tests()
        
    elif choice == "4":
        return run_examples()
        
    else:
        logger.error("无效选择")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        logger.error(f"运行时出错: {e}")
        sys.exit(1)
