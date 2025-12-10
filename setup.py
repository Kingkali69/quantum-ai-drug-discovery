#!/usr/bin/env python3
"""
Setup script for Quantum-AI Drug Discovery Framework
Built by KK&GDevOps LLC - Enterprise-grade packaging
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum-ai-drug-discovery",
    version="1.0.0",
    author="KK&GDevOps LLC",
    author_email="contact@kkgdevops.com",
    description="Revolutionary Quantum-AI Drug Discovery Framework with hybrid quantum-classical neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-ai-drug-discovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch-scatter>=2.1.0",
            "torch-sparse>=0.6.0",
            "torch-cluster>=1.6.0",
            "torch-spline-conv>=1.2.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quantum-drug-discovery=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.sql"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/quantum-ai-drug-discovery/issues",
        "Source": "https://github.com/yourusername/quantum-ai-drug-discovery",
        "Documentation": "https://quantum-ai-drug-discovery.readthedocs.io/",
    },
    keywords=[
        "quantum-computing",
        "drug-discovery",
        "molecular-property-prediction",
        "graph-neural-networks",
        "qiskit",
        "pytorch",
        "artificial-intelligence",
        "machine-learning",
        "quantum-machine-learning",
        "pharmaceutical-research",
        "cheminformatics",
        "rdkit",
    ],
)
