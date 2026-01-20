from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# 读取requirements
def read_requirements():
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "peft>=0.4.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0",
    ]
    return requirements

setup(
    name="dual-lora-federated",
    version="1.0.0",
    description="Dual-LoRA for Federated Foundation Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="FedSA-LoRA Team",
    author_email="",
    url="https://github.com/your-repo/dual-lora-federated",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dual-lora-train=run_dual_lora:main",
            "dual-lora-test=test_dual_lora:run_all_tests",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md"],
    },
)
