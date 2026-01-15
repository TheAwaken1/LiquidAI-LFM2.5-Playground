# Liquid LFM2.5 Playground

A Pinokio app for easily running LFM2.5-Audio-1.5B – Real-time voice chat, text-to-speech, and long-form speech-to-text (ASR). locally with a Gradio web UI & LFM2.5-VL-1.6B – Multi-image vision-language reasoning (describe, OCR, compare, reason step-by-step).

### Features
- One-click installation with automatic virtual environment and dependency setup
- GPU-accelerated PyTorch (with platform-specific torch builds via `torch.js`)
- LFM2.5-Audio-1.5B – Real-time voice chat, text-to-speech, and long-form speech-to-text (ASR).
- LFM2.5-VL-1.6B – Multi-image vision-language reasoning (describe, OCR, compare, reason step-by-step).
- Tested on Nvidia's 5080 GPU, was able to run it on NVidia 3060 but had some pause/delay in the speech-to-speech tab, all other tabs worked properly on the 3060 (working on getting a fix)

### Credits
- Based on the `liquid-audio` Python package and its demo extras
- Pinokio scripts adapted from common community patterns
