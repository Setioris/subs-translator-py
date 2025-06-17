# Subtitle Translator 🌐🎬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/srt?label=srt%20lib)](https://pypi.org/project/srt/)
[![PyPI Version](https://img.shields.io/pypi/v/pysubs2?label=pysubs2%20lib)](https://pypi.org/project/pysubs2/)

A professional-grade tool for translating subtitle files (`.srt`, `.ass`, `.ssa`) between languages. Supports both CPU and GPU processing with automatic device detection.

## Features ✨

- **Professional Parsing** - Uses `srt` and `pysubs2` libraries for perfect format handling
- **GPU/CPU Support** - Automatic device detection with manual override
- **Tag Preservation** - Maintains all formatting tags during translation
- **BOM Handling** - Properly processes files with byte order marks
- **Multi-core CPU** - Optimized for AMD Ryzen and Intel Core processors
- **Batch Processing** - Efficient translation with automatic fallback
- **Progress Tracking** - Real-time progress bars with tqdm

## Installation 📦

### Ubuntu/Linux
```bash
# Install system dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv translator_env
source translator_env/bin/activate

# Install Python packages
pip install torch transformers tqdm srt pysubs2
```
### Windows
```bash
# Create virtual environment
python -m venv translator_env
translator_env\Scripts\Activate.ps1

# Install Python packages
pip install torch transformers tqdm srt pysubs2
```
