# 🌍 Professional Subtitle Translator using NLLB Models

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-orange)](https://huggingface.co/facebook)

A high-performance subtitle translation tool using Facebook's NLLB (No Language Left Behind) models. Supports translation between 16+ languages while preserving subtitle formatting. Perfect for translating movies, TV shows, and video content.

![Subtitle Translation Demo](demo.gif)

## ✨ Key Features

- 🌍 **Multi-language Support**: Translate between 16+ languages
- ⚡ **GPU Acceleration**: CUDA support for fast translations
- 📚 **Format Preservation**: Maintains original formatting for SRT, ASS, and SSA files
- 🤖 **AI-Powered**: Uses state-of-the-art NLLB models from Facebook Research
- ⚙️ **Optimized Performance**: Batch processing and CPU threading options
- 📊 **Two Model Sizes**: Choose between faster 600M model or higher-quality 3.3B model

## 🌐 Supported Languages

| Language | Code | NLLB Code |
|----------|------|-----------|
| English | en | eng_Latn |
| Romanian | ro | ron_Latn |
| Japanese | ja | jpn_Jpan |
| French | fr | fra_Latn |
| German | de | deu_Latn |
| Spanish | es | spa_Latn |
| Italian | it | ita_Latn |
| Russian | ru | rus_Cyrl |
| Chinese | zh | zho_Hans |
| Korean | ko | kor_Hang |
| Arabic | ar | arb_Arab |
| Hindi | hi | hin_Deva |
| Portuguese | pt | por_Latn |
| Turkish | tr | tur_Latn |
| Dutch | nl | nld_Latn |

> Run `python subtitle_translator.py --list-languages` to see all supported languages

## 💻 Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing if needed)

```bash
# Install FFmpeg (Ubuntu/Debian)
sudo apt update && sudo apt install ffmpeg
```
# Install FFmpeg (macOS)
brew install ffmpeg
```
# Install FFmpeg (Windows)
choco install ffmpeg or [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) | [Tutorial](https://www.youtube.com/watch?v=6sim9aF3g2c)
```

# Create and activate virtual environment - delete
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
deactivate
cd ...
rmdir /s /q whisper_env

```

```bash
pip install --use-pep517 -U torch transformers huggingface-hub tqdm srt pysubs2 psutil
pip install hf_xet
pip install --upgrade transformers
```
# Test
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.cuda.is_available()}'); from transformers import __version__; print(f'Transformers: {__version__}')"
```
# 🛠 Troubleshooting
Model Loading Issues:

```bash
# Clear Hugging Face cache
rm -r ~/.cache/huggingface

CUDA Out of Memory:

# Reduce batch size in code (look for self.batch_size)
# Use smaller model
python ... --model-size 600M

Encoding Problems:

# Convert to UTF-8
iconv -f ORIGINAL_ENCODING -t UTF-8 input.srt > fixed.srt

```
# Commands:

```bash

python subtitle_translator.py english_sub.srt \
    --source en --target ro \
    --model-size 600M \
    --quality balanced

python subtitle_translator.py --list-languages

```

# ❤️ Thanks alot:
- [huggingface.co](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [facebookresearch TEAM](https://github.com/facebookresearch/fairseq)
