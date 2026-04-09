from setuptools import setup, find_packages

setup(
    name="taqg",
    version="2.0.0",
    author="Raviteja Nekkalapu",
    author_email="",
    description="Type-Aware Quantization Gap: Phase-aware KV cache quantization for reasoning LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/myProjectsRavi/taqg-kv-cache-optimization",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
