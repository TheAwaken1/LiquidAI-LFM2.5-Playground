# Liquid LFM2.5 Playground

A Pinokio app for easily running Audio, VL models, and [liquid-audio](https://pypi.org/project/liquid-audio/) demo locally with a Gradio web UI.

### Features
- One-click installation with automatic virtual environment and dependency setup
- GPU-accelerated PyTorch (with platform-specific torch builds via `torch.js`)
- LFM2.5-Audio-1.5B – Real-time voice chat, text-to-speech, and long-form speech-to-text (ASR).
- LFM2.5-VL-1.6B – Multi-image vision-language reasoning (describe, OCR, compare, reason step-by-step).
- Tested on Nvidia's 5080 GPU, was able to run it on NVidia 3060 but had some pause/delay in the speech-to-speech tab, all other tabs worked properly on the 3060 (working on getting a fix)

### Installation
1. Open Pinokio
2. Go to **Discover** → Click the **+** button (or "Install from URL")
3. Paste your repo URL: `https://github.com/yourusername/your-repo-name.git`
4. Click Install → Wait for completion → Click Start

Once running, the web UI will auto-open (or check the terminal tab for the local URL, usually http://127.0.0.1:7860).

### Credits
- Based on the `liquid-audio` Python package and its demo extras
- Pinokio scripts adapted from common community patterns
