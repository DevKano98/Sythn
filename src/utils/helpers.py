"""Utility helpers for SynthID Remover"""
import os
import sys
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional
try:
    import psutil
except ImportError:
    psutil = None


def setup_logging(log_file: str = "synthid-remover.log", level: int = logging.INFO):
    """Configure application logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a")
        ]
    )
    return logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "ram_gb": 0.0,
        "ram_available_gb": 0.0,
    }

    if psutil is not None:
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / (1024**3), 1)
        info["ram_available_gb"] = round(vm.available / (1024**3), 1)

    # GPU info
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["gpu_memory_gb"] = [
                round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["cuda_available"] = False

    return info


def format_bytes(size_bytes: int) -> str:
    """Format byte size to human readable."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size(path: str) -> Optional[int]:
    """Get file size in bytes, or None if not found."""
    try:
        return Path(path).stat().st_size
    except (FileNotFoundError, OSError):
        return None


def is_valid_image(path: str) -> bool:
    """Check if file is a valid image."""
    valid_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    return Path(path).suffix.lower() in valid_ext


def get_image_files(directory: str) -> list:
    """Get all image files in directory."""
    valid_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [str(f) for f in Path(directory).iterdir() if f.suffix.lower() in valid_ext]


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename."""
    invalid = '<>:"/\\|?*'
    for char in invalid:
        name = name.replace(char, '_')
    return name.strip()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    if psutil is None:
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "percent": 0.0,
        }
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024**2), 1),
        "vms_mb": round(mem.vms / (1024**2), 1),
        "percent": process.memory_percent(),
    }
