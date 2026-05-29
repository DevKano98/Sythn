#!/usr/bin/env python3
"""SynthID Remover entry point."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import APP_NAME, APP_VERSION, COMFY_BASE_URL, COMFYUI_DIR, OUTPUT_DIR
from src.comfy import ComfyClient, ComfyManager, ComfyModelManager, ComfyPipeline, PipelineConfig
from src.utils.helpers import get_image_files, get_system_info, setup_logging
from src.utils.validators import validate_image_path


def check_system() -> int:
    print("=" * 70)
    print(f"{APP_NAME} v{APP_VERSION} - System Check")
    print("=" * 70)

    info = get_system_info()
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"CPU cores: {info['cpu_count']}")
    print(f"RAM: {info['ram_gb']} GB total, {info['ram_available_gb']} GB available")
    print(f"CUDA: {'Yes' if info.get('cuda_available') else 'No'}")
    print()

    manager = ComfyManager()
    client = ComfyClient(COMFY_BASE_URL)
    print("ComfyUI:")
    print(f"  Directory: {COMFYUI_DIR}")
    print(f"  Installed: {'Yes' if manager.is_installed() else 'No'}")
    print(f"  API ready: {'Yes' if client.is_ready() else 'No'} ({COMFY_BASE_URL})")
    print()

    models = ComfyModelManager()
    present, missing = models.validate()
    print("Model Status:")
    for name in present:
        path = models.model_path(name)
        print(f"  [OK]      {name:16s} {path}")
    for name in missing:
        print(f"  [MISSING] {name:16s} {models.model_path(name)}")
    print()

    if not manager.is_installed() or missing:
        print("Setup required. See README.md or run: python download_models.py --install-comfy")
        return 1
    print("All required local files are present.")
    return 0


def process_one(pipeline: ComfyPipeline, image_path: Path, output_dir: Path, args) -> bool:
    validate_image_path(str(image_path))
    config = PipelineConfig(
        image_path=str(image_path),
        positive_prompt=args.positive,
        negative_prompt=args.negative,
        denoise_auto=args.denoise_auto,
        denoise_manual=args.denoise,
        face_enhancement=not args.no_faces,
        steps=args.steps,
        seed=args.seed,
        output_format=args.format,
        output_quality=args.quality,
    )

    def progress(stage, total, message):
        print(f"[{stage}/{total}] {image_path.name}: {message}")

    result = pipeline.run(config, progress)
    if result.success:
        print(f"Saved: {result.output_path}")
        return True
    print(f"Failed: {result.error_message}", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    parser.add_argument("--dev", action="store_true", help="Enable debug logging")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--input", type=str, help="Input image or folder")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output folder")
    parser.add_argument("--check", action="store_true", help="Check ComfyUI, models, and system")
    parser.add_argument("--positive", default="high quality photo, detailed, sharp focus")
    parser.add_argument("--negative", default="watermark, synthid, text, logo, blurry, low quality, artifact")
    parser.add_argument("--denoise", type=float, default=0.20)
    parser.add_argument("--denoise-auto", action="store_true")
    parser.add_argument("--no-faces", action="store_true")
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--format", choices=["png", "jpg", "webp"], default="png")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.dev else logging.INFO)

    if args.check:
        return check_system()

    if args.headless or args.input:
        if not args.input:
            print("Headless mode requires --input", file=sys.stderr)
            return 1
        input_path = Path(args.input)
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = ComfyPipeline(output_dir=str(output_dir))
        try:
            if input_path.is_file():
                return 0 if process_one(pipeline, input_path, output_dir, args) else 1
            if input_path.is_dir():
                files = [Path(p) for p in get_image_files(str(input_path))]
                if not files:
                    print(f"No images found in {input_path}", file=sys.stderr)
                    return 1
                ok = 0
                for file_path in files:
                    ok += 1 if process_one(pipeline, file_path, output_dir, args) else 0
                print(f"Batch complete: {ok}/{len(files)} successful")
                return 0 if ok == len(files) else 1
            print(f"Invalid input: {input_path}", file=sys.stderr)
            return 1
        finally:
            pipeline.stop()

    from src.gui.app import main as gui_main
    gui_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
