# SynthID Remover v2.0

Python desktop and CLI wrapper for running a ComfyUI SynthID-removal workflow.

This app does not reimplement ComfyUI internals. It manages a ComfyUI subprocess, uploads images through ComfyUI's REST API, queues workflow API JSON, polls completion from `/history/{prompt_id}`, downloads the output through `/view`, and displays a before/after comparison.

## Setup

1. Create and activate a Python environment.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Install ComfyUI.

Automatic:

```bash
python download_models.py --install-comfy
```

Manual:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI
cd ComfyUI
pip install -r requirements.txt
cd ..
```

If ComfyUI is elsewhere, set `COMFYUI_DIR` before launching this app.

3. Download models into ComfyUI.

```bash
python download_models.py
```

Expected paths:

```text
ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors
ComfyUI/models/text_encoders/qwen_3_4b.safetensors
ComfyUI/models/vae/ae.safetensors
ComfyUI/models/controlnet/Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors
ComfyUI/models/ultralytics/yolov8n-face.pt
```

4. Install the ComfyUI custom nodes required by your working workflow:

- Z-Image model loader support if your ComfyUI build does not already include it
- ControlNet preprocessor node providing `CannyEdgePreprocessor` or `Canny`
- Face/inpaint nodes providing `UltralyticsDetectorProvider`, `BboxDetectorForEach`, `InpaintCrop`, and `InpaintStitch`

The app validates required node class names through `/object_info` before queueing.

## Run

GUI:

```bash
python main.py
```

CLI:

```bash
python main.py --headless --input image.png
python main.py --headless --input ./folder --output ./results
python main.py --check
```

## Workflow

The generated API workflow follows this structure:

- Load Image -> VAE Encode
- Load CLIP `qwen_3_4b.safetensors` -> positive/negative CLIP Text Encode
- Load Diffusion Model `z_image_turbo_bf16.safetensors`
- Load ControlNet -> Canny preprocessing -> Apply ControlNet
- KSampler: Euler/simple, default steps 9, cfg 1.0, denoise 0.2
- VAE Decode -> Save Image
- Optional face pass: YOLO face detector -> bbox detector -> inpaint crop -> low-denoise KSampler -> inpaint stitch -> Save Image

Output format and quality are applied after ComfyUI returns the generated image.
"# Sythn" 
