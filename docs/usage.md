# Usage Guide

## Quick Start

### 1. Launch the Application

```bash
python main.py
```

### 2. Load an Image

- **Click "📁 Load Image"** to browse for an image
- **Drag and drop** an image directly onto the preview area
- Supported formats: PNG, JPG, JPEG, WEBP, BMP

### 3. Configure Settings

#### Prompts
- **Positive prompt**: Describe what you want in the output (e.g., "high quality photo, detailed")
- **Negative prompt**: Describe what to avoid (e.g., "watermark, synthid, text, logo")

#### Denoise Settings
- **Auto-calculate**: Automatically determines denoise strength based on image resolution
  - 512x512 → ~0.08
  - 1024x1024 → ~0.12
  - 2048x2048 → ~0.15
- **Manual**: Override with custom value (0.0 - 1.0)
  - Lower = more similar to original
  - Higher = more reconstruction

#### Face Enhancement
- **Enable**: Detects and enhances faces separately
- **Face denoise scale**: Multiplier for face-specific denoise (typically 0.5-0.8 of main denoise)

#### Generation Settings
- **Steps**: Number of diffusion steps (4 recommended with Lightning LoRA)
- **Seed**: Random seed (-1 for random, fixed for reproducible results)

#### Output Settings
- **Format**: PNG, JPG, or WEBP
- **Quality**: 1-100 (for lossy formats)

### 4. Process

Click **"🚀 Remove SynthID"** to start processing.

The pipeline runs through 12 stages:
1. Image loading & normalization
2. Adaptive denoise calculation
3. Edge extraction (Canny)
4. ControlNet conditioning
5. Prompt encoding
6. Latent encoding
7. Main reconstruction
8. Face detection
9. Face segmentation
10. Face reconstruction
11. Compositing
12. Export

### 5. Review Results

- Switch to the **"Before/After"** tab to compare
- Use the slider to reveal the before/after split
- Output is automatically saved to `output/`

## Batch Processing

Process entire folders without GUI:

```bash
# Single image
python main.py --headless --input image.png

# Entire folder
python main.py --headless --input ./photos/ --output ./results/
```

## Building an Executable

```bash
pip install pyinstaller
python build.py
```

The executable will be created in `dist/SynthID-Remover/`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing models" warning | Run `python download_models.py` or manually download models to `models/` |
| Out of memory | Reduce image size, use CPU mode, or close other applications |
| Slow processing | Ensure GPU is available (check status bar), use 4 steps with Lightning LoRA |
| Face detection not working | Verify `yolov8n-face.pt` exists in `models/` |
| GUI won't start | Check Python 3.10+ and install requirements: `pip install -r requirements.txt` |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Load image |
| Ctrl+S | Open output folder |
| Ctrl+R | Start processing |
| Escape | Cancel processing |
