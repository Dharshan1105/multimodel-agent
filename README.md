# Multimodal Agentic AI Assistant

A locally running multimodal AI assistant built with LangGraph, Ollama, 
Gemma3, LLaVA, Stable Diffusion, and Kokoro TTS.

## What It Does

- Text conversation with memory across turns
- Spoken responses using Kokoro TTS
- Image understanding using LLaVA
- Image generation using Stable Diffusion

Everything runs locally. No cloud API calls.

## Requirements

- Python 3.12
- Ollama installed and running
- NVIDIA GPU recommended (tested on 5.65 GB VRAM)

## Setup

### 1. Install system dependencies
sudo apt-get install portaudio19-dev espeak-ng

### 2. Pull Ollama models
ollama pull gemma3
ollama pull llava

### 3. Clone the repo
git clone https://github.com/yourusername/multimodel-agent.git
cd multimodel-agent

### 4. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

### 5. Install Python dependencies
pip install -r requirements.txt

### 6. Download Kokoro model files
Place these two files in the project root:
- kokoro-v0_19.onnx
- voices.bin

Download from:
https://github.com/thewh1teagle/kokoro-onnx/releases

### 7. Run
python3 main.py

## Project Structure

multimodel-agent/
├── main.py                    # Entry point
├── router.py                  # LangGraph workflow
├── state.py                   # Shared state
├── nodes/                     # One node per capability
├── models/                    # One client per AI model
├── memory/                    # Memory stores
├── outputs/                   # Generated images saved here
└── uploads/                   # Place images here for vision mode

## Usage

Select a mode when prompted:
- text    : text conversation
- audio   : spoken responses
- vision  : describe an image
- imagine : generate an image from text

Type switch to change modes. Type exit to quit.

## Owner
Dharshan
