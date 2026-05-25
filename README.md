# msLoRA: Modality-Specific Adaptation for Multimodal Large Language Models
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

msLoRA is an efficient fine-tuning strategy for Multimodal Large Language Models (MLLMs). It introduces **Modality-Specific Adapters** that decouple the adaptation processes of text, image, and audio. This architecture addresses the "modality interference" problem in standard LoRA, enabling better cross-modal alignment with a significantly reduced parameter budget.

## 🚀 Core Features

- **Modality-Specific Decoupling**: Separate low-rank matrices for textual, visual, and auditory updates to eliminate gradient interference.
- **Asymmetric Rank Allocation**: Supports independent rank scaling (e.g., higher ranks for high-entropy visual features) to mitigate textual dominance.
- **Cumulative Information Span**: Theoretically and empirically proven to capture a broader functional subspace by decentralizing spectral energy.
- **Halved Optimal Rank**: Matches or exceeds standard LoRA performance using only ~50% of the rank budget.

## 📂 Project Structure

FileDescriptionmodule.pyCore Architecture: Implementation of MyLoraLayer and the msLoRA wrapper.main.pyTraining Pipeline: Entry point for training across tasks (MSR-VTT, ScienceQA, etc.).analysis.pyDiagnostic Tools: Scripts for SVD spectral analysis and gradient orthogonality verification.utils.pyData Engine: Multimodal processors (CLIP, Wav2Vec2) and task-specific loaders.train.shAutomation: Bash script for streamlined experiment execution.

## 🛠️ Installation

Prerequisites
- Python 3.10+
- CUDA-enabled GPU (e.g., RTX 3090, RTX 6000 Ada)

```bash
git clone https://github.com
cd msLoRA
pip install -r requirements.txt
```

## 📖 Usage

### 1. Download Datasets
- **MVSA_Single**: [xwycyj/MVSA-Single](https://huggingface.co/xwycyj/MVSA-Single)
- **MSR-VTT**: [VLM2Vec/MSR-VTT](https://huggingface.co/VLM2Vec/MSR-VTT)
### 2. Pre-trained Models

1). Download the pretrained models (Swin, BERT, Wav2Vec) Hugging Face

- **CLIP**: [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- **Wav2Vec**: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- **Qwen2.5-7B**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

### 3. Quick Start (Training)
```bash
./shell/train.sh
```
### 4. SVD & Gradient Analysis
   To verify the modality independence and spectral distribution:
```
analysis.ipynb
```

## 📝 Citation

If you find this work useful in your research, please consider citing:
```

```
