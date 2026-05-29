# Changelog

All notable changes to SynthID Remover will be documented in this file.

## [2.0.0] - 2026-05-29

### Added
- Standalone desktop application (no ComfyUI required)
- 12-stage processing pipeline with full architecture documentation
- Adaptive denoise calculation based on image resolution
- Canny edge extraction with ControlNet structural guidance
- Qwen2.5-VL prompt encoding
- Main reconstruction using Qwen Image 2512 GGUF model
- YOLOv8 face detection
- MediaPipe + SAM face segmentation
- Face-specific reconstruction with z_image_turbo
- Feathered compositing for seamless face merging
- Modern CustomTkinter GUI with dark theme
- Drag-and-drop image loading
- Before/After comparison slider
- Real-time stage-by-stage progress tracking
- Model status widget showing availability
- Batch processing via headless CLI mode
- PyInstaller build script for executable creation
- Comprehensive test suite
- Model download helper script
- Cross-platform support (Windows, macOS, Linux)

### Architecture
- Modular stage-based pipeline design
- ModelManager for lazy loading and caching
- GGUF inference via llama-cpp-python
- VAE encode/decode for latent space operations
- ControlNet conditioning for structure preservation
- DiffusionSampler with CFG and Lightning LoRA support

## [1.0.0] - 2026-01-15

### Added
- Initial ComfyUI workflow-based implementation
- Custom nodes for adaptive denoise and face detailer
- Basic image reconstruction pipeline
