"""Setup script for SynthID Remover"""
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="synthid-remover",
    version="2.0.0",
    author="SynthID Remover Contributors",
    description="Standalone desktop app for removing SynthID watermarks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/synthid-remover",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "llama-cpp-python>=0.2.0",
        "diffusers>=0.25.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0",
        "segment-anything>=1.0",
        "customtkinter>=5.2.2",
        "Pillow>=10.0.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "packaging>=23.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "synthid-remover=main:main",
        ],
    },
)
