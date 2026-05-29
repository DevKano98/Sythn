#!/usr/bin/env python3
"""Install/check ComfyUI and download required SynthID Remover models."""
import argparse
import sys

from config import COMFYUI_DIR, MODEL_SPECS
from src.comfy import ComfyManager, ComfyModelManager


def print_status(models: ComfyModelManager) -> bool:
    present, missing = models.validate()
    print("Model Status")
    print("-" * 80)
    for name in present:
        print(f"[OK]      {name:16s} {models.model_path(name)}")
    for name in missing:
        print(f"[MISSING] {name:16s} {models.model_path(name)}")
    print("-" * 80)
    return not missing


def main() -> int:
    parser = argparse.ArgumentParser(description="SynthID Remover ComfyUI setup")
    parser.add_argument("--check", action="store_true", help="Only check ComfyUI and model files")
    parser.add_argument("--force", action="store_true", help="Re-download existing model files")
    parser.add_argument("--install-comfy", action="store_true", help="Install ComfyUI with comfy-cli")
    args = parser.parse_args()

    print(f"ComfyUI directory: {COMFYUI_DIR}")
    manager = ComfyManager()
    if args.install_comfy and not manager.is_installed():
        print("Installing ComfyUI with comfy-cli...")
        manager.install_with_comfy_cli()

    print(f"ComfyUI installed: {'Yes' if manager.is_installed() else 'No'}")
    print()

    models = ComfyModelManager()
    models.ensure_dirs()
    all_present = print_status(models)
    if args.check:
        return 0 if manager.is_installed() and all_present else 1

    if not manager.is_installed():
        print("ComfyUI is not installed. Run: python download_models.py --install-comfy")
        return 1

    missing = models.missing_models
    if not missing and not args.force:
        print("All models already present.")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is required. Run: pip install huggingface-hub")
        return 1

    print("Downloading models:")
    for name, spec in MODEL_SPECS.items():
        if name in missing or args.force:
            print(f"  {name}: {spec['repo_id']} / {spec['filename']}")
    models.download_missing(force=args.force)
    print()
    print_status(models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
