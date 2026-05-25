# msLoRA: Modality-Specific Adaptation for Multimodal Large Language Models
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

msLoRA is an efficient fine-tuning strategy for Multimodal Large Language Models (MLLMs). It introduces **Modality-Specific Adapters** that decouple the adaptation processes of text, image, and audio. This architecture addresses the "modality interference" problem in standard LoRA, enabling better cross-modal alignment with a significantly reduced parameter budget.

## 🚀 Core Features

Modality-Specific Decoupling: Separate low-rank matrices for textual, visual, and auditory updates to eliminate gradient interference.Asymmetric Rank Allocation: Supports independent rank scaling (e.g., higher ranks for high-entropy visual features) to mitigate textual dominance.Cumulative Information Span: Theoretically and empirically proven to capture a broader functional subspace by decentralizing spectral energy.Halved Optimal Rank: Matches or exceeds standard LoRA performance using only ~50% of the rank budget.

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

1. Quick Start (Training)Use the provided script to start training on MSR-VTT using a Qwen2.5-7B backbone:Bashbash train.sh
2. Manual ConfigurationYou can customize the rank and scaling factors via command-line arguments:Bashpython main.py \
    -task msrvtt \
    -llm_model ./path/to/llm \
    -clip_model ./path/to/clip \
    -r 16 \
    -multimodal_scaling 4 \
    -epochs 2 \
    -batch_size 32
3. SVD & Gradient AnalysisTo verify the modality independence and spectral distribution:Bashpython analysis.py --model_path ./path/to/checkpoint

## 📝 Citation

If you find this work useful in your research, please consider citing:
