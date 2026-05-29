# Architecture Documentation

## Overview

SynthID Remover uses a 12-stage pipeline architecture designed for modular, maintainable, and extensible image processing. Each stage is self-contained with clear inputs and outputs.

## Pipeline Stages

### Stage 1: Image Input
- **Purpose**: Load and normalize input images
- **Input**: File path (PNG, JPG, WEBP, BMP)
- **Output**: Normalized RGB numpy array, metadata (original size, path)
- **Key Operations**: Format detection, color space conversion, optional resizing

### Stage 2: Adaptive Denoise
- **Purpose**: Calculate optimal denoise strength
- **Input**: Image array
- **Output**: Denoise float value, resolution info
- **Algorithm**: Logarithmic scaling based on megapixels
  - `denoise = 0.05 + 0.03 * sqrt(megapixels)`
- **Presets**: Conservative, Balanced, Aggressive

### Stage 3: Edge Extraction
- **Purpose**: Generate structural guidance map
- **Input**: RGB image
- **Output**: 3-channel edge map
- **Algorithm**: OpenCV Canny edge detector
- **Parameters**: low_threshold=50, high_threshold=150

### Stage 4: ControlNet Conditioning
- **Purpose**: Inject structural constraints into diffusion
- **Input**: Edge map, prompt embeddings
- **Output**: Conditioning tensor
- **Model**: qwen_image_canny_diffsynth_controlnet.safetensors

### Stage 5: Prompt Encoding
- **Purpose**: Convert text to latent guidance vectors
- **Input**: Positive/negative prompts
- **Output**: Embedding tensors
- **Model**: Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf

### Stage 6: Latent Encoding
- **Purpose**: Convert image to latent space
- **Input**: RGB image tensor
- **Output**: Latent tensor (typically 1/8 spatial resolution)
- **Model**: qwen_image_vae.safetensors

### Stage 7: Main Reconstruction
- **Purpose**: Core diffusion-based image reconstruction
- **Input**: Latent, conditioning, denoise value
- **Output**: Reconstructed latent
- **Model**: qwen-image-2512-Q4_K_M.gguf
- **Accelerator**: Lightning LoRA (4 steps)
- **Method**: Latent diffusion with classifier-free guidance

### Stage 8: Face Detection
- **Purpose**: Locate faces in image
- **Input**: Original image
- **Output**: Bounding boxes list
- **Model**: YOLOv8n (yolov8n-face.pt)
- **Post-processing**: 20% padding around detections

### Stage 9: Face Segmentation
- **Purpose**: Create precise face masks
- **Input**: Image, bounding boxes
- **Output**: Binary masks, face region crops
- **Method**: Bbox-based mask with Gaussian feathering (radius=20px)

### Stage 10: Face Reconstruction
- **Purpose**: Face-specific enhancement
- **Input**: Face crops
- **Output**: Enhanced face regions
- **Model**: z_image_turbo-Q4_K_M.gguf
- **Strategy**: Lower denoise (0.7x main) to preserve identity

### Stage 11: Compositing
- **Purpose**: Merge enhanced faces back
- **Input**: Base image, enhanced faces, masks, bboxes
- **Output**: Final composite image
- **Method**: Alpha blending with feathered mask edges

### Stage 12: Export
- **Purpose**: Save final output
- **Input**: Final image array
- **Output**: File on disk
- **Formats**: PNG (lossless), JPG (quality-adjustable), WEBP

## Data Flow

```
Image Path
  ↓
[Stage 1] Load → RGB Array
  ↓
[Stage 2] Calculate Denoise
  ↓
[Stage 3] Canny Edges
  ↓
[Stage 4] ControlNet Conditioning ← Edges
  ↓
[Stage 5] Prompt Encoding ← Text Prompts
  ↓
[Stage 6] VAE Encode → Latent
  ↓
[Stage 7] Diffusion Reconstruction ← Latent + Conditioning + Denoise
  ↓
[Stage 8] Face Detection ← Original Image
  ↓
[Stage 9] Face Segmentation ← Bboxes
  ↓
[Stage 10] Face Reconstruction ← Face Crops
  ↓
[Stage 11] Compositing ← Base + Faces + Masks
  ↓
[Stage 12] Export → File
```

## Model Architecture

### ModelManager
- Lazy loading: models loaded only when first requested
- Caching: loaded models kept in memory
- Unloading: explicit memory management for large models

### Inference Wrappers
- **GGUFInference**: llama-cpp-python wrapper for quantized models
- **VAEInference**: Safetensors-based VAE encode/decode
- **ControlNetInference**: Structural conditioning application
- **DiffusionSampler**: Sampling loop with CFG and schedulers
- **FaceEnhancer**: Face-specific lower-denoise reconstruction

## Extensibility

### Adding a New Stage
1. Create class in `src/pipeline/stages.py`
2. Add to `SynthIDPipeline._setup_stages()`
3. Add execution in `SynthIDPipeline.run()`
4. Update progress callback

### Adding a New Model
1. Add entry to `config.py` MODELS dict
2. Add loader method in `ModelManager`
3. Add inference wrapper in `src/models/inference.py`
4. Update model status widget

## Performance Considerations

| Stage | GPU Usage | Memory | Bottleneck |
|-------|-----------|--------|------------|
| Stage 7 | High | 4-8GB | Main reconstruction |
| Stage 10 | High | 2-4GB | Face reconstruction |
| Stage 6 | Medium | 1GB | VAE encode/decode |
| Stage 8 | Low | 500MB | Face detection |
| Others | Minimal | <100MB | CPU-bound |

## Threading Model

- GUI runs on main thread
- Pipeline runs on background thread
- Progress updates via `tkinter.after()` thread-safe callbacks
- Cancellation via flag check between stages
