# Model Documentation

## Required Models

All models must be placed in the `models/` directory.

### Main Reconstruction

**File**: `qwen-image-2512-Q4_K_M.gguf`
**Size**: ~4 GB
**Purpose**: Primary diffusion model for image reconstruction
**Format**: GGUF (Q4_K_M quantization)
**Architecture**: Qwen Image 2512
**Role**: Stage 7 - Main Reconstruction

This is the core model that performs the latent-space diffusion to reconstruct the image without the SynthID watermark. It uses 4-bit quantization for efficient inference.

### VAE

**File**: `qwen_image_vae.safetensors`
**Size**: ~300 MB
**Purpose**: Image ↔ Latent space conversion
**Format**: Safetensors
**Role**: Stage 6 (Encode), Stage 7 (Decode)

The Variational Autoencoder compresses images into a latent representation (typically 1/8th spatial resolution) and decompresses them back. This is what makes diffusion models efficient.

### ControlNet

**File**: `qwen_image_canny_diffsynth_controlnet.safetensors`
**Size**: ~1.2 GB
**Purpose**: Structural guidance
**Format**: Safetensors
**Role**: Stage 4 - ControlNet Conditioning

ControlNet takes the Canny edge map from Stage 3 and injects structural constraints into the diffusion process. Without it, the reconstructed image might drift significantly from the original composition.

### Prompt Encoder

**File**: `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
**Size**: ~4 GB
**Purpose**: Text-to-embedding conversion
**Format**: GGUF (Q4_K_M quantization)
**Architecture**: Qwen2.5-VL-7B-Instruct
**Role**: Stage 5 - Prompt Encoding

Converts positive and negative prompts into latent guidance vectors that steer the diffusion model toward the desired output.

### Lightning LoRA

**File**: `Qwen-Image-2512-Lightning-4steps-V1.0.safetensors`
**Size**: ~150 MB
**Purpose**: Accelerated sampling
**Format**: Safetensors (LoRA)
**Role**: Stage 7 - Main Reconstruction (accelerator)

A Low-Rank Adaptation that reduces the required sampling steps from 20-50 down to 4 while maintaining quality. Essential for fast processing.

### Face Detection

**File**: `yolov8n-face.pt`
**Size**: ~6 MB
**Purpose**: Face localization
**Format**: PyTorch
**Architecture**: YOLOv8 nano
**Role**: Stage 8 - Face Detection

Lightweight and fast face detection. Runs on CPU efficiently. Detects faces with bounding boxes and applies 20% padding for better context.

### Face Reconstruction

**File**: `z_image_turbo-Q4_K_M.gguf`
**Size**: ~2 GB
**Purpose**: Face-specific enhancement
**Format**: GGUF (Q4_K_M quantization)
**Role**: Stage 10 - Face Reconstruction

A specialized model for enhancing face regions. Uses lower denoise strength than the main model to preserve facial identity while removing artifacts.

### Face Encoder

**File**: `Qwen_3_4b-imatrix-IQ4_XS.gguf`
**Size**: ~2 GB
**Purpose**: Face prompt encoding
**Format**: GGUF (IQ4_XS quantization)
**Architecture**: Qwen 3B
**Role**: Stage 10 - Face Reconstruction (prompt encoding)

Provides text conditioning for the face reconstruction model, allowing face-specific prompts like "detailed face, high quality".

### Face VAE

**File**: `ae.safetensors`
**Size**: ~80 MB
**Purpose**: Face latent conversion
**Format**: Safetensors
**Role**: Stage 10 - Face Reconstruction (VAE)

Smaller VAE optimized for face-sized inputs (typically 256x256 to 512x512 crops).

## Model Loading Strategy

The app uses **lazy loading**:
1. Models are only loaded when first needed
2. Once loaded, they stay in memory (cached)
3. You can manually unload models to free memory
4. On app close, all models are unloaded

## Memory Requirements

| Configuration | VRAM Required | RAM Required |
|---------------|-------------|--------------|
| GPU mode, all models loaded | 8-12 GB | 16 GB |
| GPU mode, single image | 6-8 GB | 12 GB |
| CPU mode | 0 GB | 16-24 GB |
| CPU mode, quantized | 0 GB | 12-16 GB |

## Download Sources

Models are available from:
- **HuggingFace**: Primary source for Qwen models
- **Ultralytics**: YOLOv8 face detection
- **CivitAI**: Some LoRA models
- **ModelScope**: Alternative mirror for Qwen models

## Verification

After downloading, verify models with:
```bash
python main.py --check
```

This checks file existence and reports sizes.
