# Frequently Asked Questions

## General

### What is SynthID?
SynthID is an invisible watermarking technology developed by Google DeepMind. It embeds imperceptible signals into AI-generated images to identify them as synthetic.

### Is this tool legal to use?
This tool is intended for legitimate purposes such as:
- Removing watermarks from images you own
- Testing watermark robustness for research
- Educational purposes

Always respect copyright and terms of service.

### Does this work on all AI-generated images?
The tool is specifically designed for images with SynthID watermarks. It uses diffusion-based reconstruction which may also affect other types of watermarks, but effectiveness varies.

## Installation

### Which Python version do I need?
Python 3.10 or newer is required. Python 3.11 is recommended.

### Do I need a GPU?
No, but a CUDA-capable GPU (NVIDIA) significantly speeds up processing. CPU mode works but is slower.

### How much disk space do I need?
- Application: ~50 MB
- Models: ~15-20 GB total
- Working space: ~5 GB recommended

### Can I use this on macOS?
Yes, but GPU acceleration is limited. M1/M2 Macs can use MPS backend with some modifications.

## Usage

### Why is processing slow?
Check these factors:
1. **GPU availability**: Look at the status bar. If it shows "CPU", processing will be slow.
2. **Image size**: Larger images take longer. The app auto-resizes to 2048px max.
3. **Step count**: Use 4 steps with Lightning LoRA for fastest results.
4. **Model loading**: First run loads models into memory. Subsequent runs are faster.

### What denoise strength should I use?
- **Auto mode** (recommended): The app calculates based on resolution
- **Manual 0.05-0.10**: Conservative, very similar to original
- **Manual 0.10-0.15**: Balanced, good watermark removal
- **Manual 0.15-0.25**: Aggressive, more reconstruction

### Will face enhancement preserve identity?
Yes, face enhancement uses 0.7x the main denoise strength to preserve facial features while removing artifacts.

### Can I process multiple images at once?
Yes, use headless batch mode:
```bash
python main.py --headless --input ./folder/ --output ./results/
```

## Models

### Where do I download the models?
Run the helper script:
```bash
python download_models.py
```

Or download manually from HuggingFace and place in `models/`.

### Can I use different models?
The pipeline is designed for specific Qwen-based models. Using different models would require code modifications.

### Why are models so large?
GGUF quantized models are already compressed. The original full-precision models would be 2-4x larger.

## Troubleshooting

### "Missing models" warning
Download the required model files to the `models/` folder. See the README for the full list.

### Out of memory error
1. Close other applications
2. Process smaller images
3. Use CPU mode (slower but uses less VRAM)
4. Enable model unloading in settings

### Face detection not working
Ensure `yolov8n-face.pt` is in the `models/` folder. This is a small model (~6MB) from Ultralytics.

### GUI won't start
1. Check Python version: `python --version` (need 3.10+)
2. Install requirements: `pip install -r requirements.txt`
3. Check for tkinter: `python -c "import tkinter"`
4. Try: `python main.py --dev` for debug output

### Build fails with PyInstaller
1. Update PyInstaller: `pip install --upgrade pyinstaller`
2. Try `--onedir` instead of `--onefile`
3. Check hidden imports in `build.py`

## Performance

### How long does one image take?
- **GPU + Lightning (4 steps)**: 5-15 seconds
- **GPU + Standard (20 steps)**: 30-60 seconds
- **CPU + Lightning**: 2-5 minutes
- **CPU + Standard**: 10-20 minutes

### Can I speed up batch processing?
Yes:
1. Use GPU mode
2. Use 4 steps with Lightning LoRA
3. Disable face enhancement if not needed
4. Process at lower resolution

## Development

### How do I add a new pipeline stage?
See `docs/architecture.md` for the extensibility guide.

### How do I contribute?
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Where is the ComfyUI version?
This package is the standalone app only. ComfyUI workflows are not bundled; the `workflows/` folder is intentionally unused.
