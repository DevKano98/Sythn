#!/usr/bin/env python3
"""Build standalone executable for SynthID Remover

Usage:
    python build.py              # Build for current platform
    python build.py --onefile    # Single executable (slower startup)
    python build.py --clean      # Clean build artifacts first

Requirements:
    pip install pyinstaller
"""
import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

APP_NAME = "SynthID-Remover"
VERSION = "2.0.0"

def clean_build():
    """Remove previous build artifacts."""
    dirs = ["build", "dist", "__pycache__"]
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed {d}/")

    # Clean .spec files
    for f in Path(".").glob("*.spec"):
        f.unlink()
        print(f"Removed {f}")

def build(onefile: bool = False):
    """Build executable with PyInstaller."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--windowed",  # No console window for GUI app
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Add hidden imports for packages that PyInstaller might miss
    hidden_imports = [
        "llama_cpp",
        "safetensors",
        "diffusers",
        "transformers",
        "ultralytics",
        "cv2",
        "PIL",
        "customtkinter",
        "numpy",
        "torch",
        "torchvision",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Add data files
    cmd.extend(["--add-data", "config.py:."])
    cmd.extend(["--add-data", "README.md:."])

    # Icon (if available)
    icon_path = "assets/icon.ico" if sys.platform == "win32" else "assets/icon.icns"
    if Path(icon_path).exists():
        cmd.extend(["--icon", icon_path])

    # Main entry point
    cmd.append("main.py")

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print()
        print("=" * 60)
        print("Build successful!")
        print("=" * 60)
        if onefile:
            print(f"Executable: dist/{APP_NAME}.exe (Windows) or dist/{APP_NAME} (Unix)")
        else:
            print(f"App folder: dist/{APP_NAME}/")
            print(f"Executable: dist/{APP_NAME}/{APP_NAME}.exe (Windows)")
        print()
        print("Note: Model files are NOT included in the build.")
        print("Users must place models in the models/ folder next to the executable.")
    else:
        print()
        print("Build failed!")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build SynthID Remover executable")
    parser.add_argument("--onefile", action="store_true", help="Build single-file executable")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    args = parser.parse_args()

    if args.clean:
        clean_build()

    build(onefile=args.onefile)

if __name__ == "__main__":
    main()
