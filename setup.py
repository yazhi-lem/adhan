"""Setup configuration for Adhan."""

from setuptools import setup, find_packages
from pathlib import Path

root_dir = Path(__file__).parent
readme = (root_dir / "README.md").read_text(encoding="utf-8") if (root_dir / "README.md").exists() else ""

setup(
    name="adhan",
    version="0.2.0",
    description="Tamil Large Language Model - Modular, Scalable",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Yazhi Foundation",
    author_email="info@yazhi.dev",
    url="https://github.com/yazhi-lem/adhan",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.21",
        "requests>=2.28",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "dev": ["pytest>=7.0", "pytest-cov>=3.0", "black>=22.0", "ruff>=0.0.200"],
        "transformers": ["transformers>=4.20", "datasets>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "adhan=adhan.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
