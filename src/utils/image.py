"""Image processing utilities for SynthID Remover"""
import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, Optional, Union


def load_image(path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img: np.ndarray, path: str, quality: int = 95):
    """Save numpy image to file."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
        pil_img.save(path, quality=quality)
    else:
        pil_img.save(path)


def resize_image(img: np.ndarray, max_size: int = 2048) -> np.ndarray:
    """Resize image if larger than max_size while maintaining aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def canny_edge_detect(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Extract Canny edges from image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    # Convert to 3-channel for model input
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_rgb


def numpy_to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy RGB [H,W,3] to tensor [1,3,H,W] normalized to [-1,1]."""
    img = img.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0  # [0,1] -> [-1,1]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [1,3,H,W] in [-1,1] to numpy RGB [H,W,3] in [0,255]."""
    img = tensor.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img + 1.0) / 2.0  # [-1,1] -> [0,1]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    arr = tensor_to_numpy(tensor)
    return Image.fromarray(arr)


def pil_to_tensor(pil_img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Convert PIL Image to tensor."""
    arr = np.array(pil_img.convert("RGB"))
    return numpy_to_tensor(arr, device)


def create_mask_from_bbox(img_shape: Tuple[int, ...], bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Create binary mask from bounding box."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255
    return mask


def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    """Apply Gaussian blur to mask edges for smooth compositing."""
    blurred = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
    return blurred


def composite_images(background: np.ndarray, foreground: np.ndarray, 
                     mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Composite foreground onto background using mask."""
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1

    # Resize foreground and mask to bbox size
    fg_resized = cv2.resize(foreground, (w, h), interpolation=cv2.INTER_LANCZOS4)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize mask to [0,1]
    mask_norm = mask_resized.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)

    # Blend
    roi = background[y1:y2, x1:x2].astype(np.float32)
    fg = fg_resized.astype(np.float32)
    blended = roi * (1 - mask_3ch) + fg * mask_3ch

    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def get_image_resolution_info(img: np.ndarray) -> dict:
    """Get resolution info for adaptive denoise."""
    h, w = img.shape[:2]
    mp = (w * h) / 1_000_000
    return {
        "width": w,
        "height": h,
        "megapixels": mp,
        "pixels": w * h
    }


def calculate_adaptive_denoise(img: np.ndarray, scale: float = 1.0,
                                min_val: float = 0.05, max_val: float = 0.25) -> float:
    """Calculate denoise strength based on resolution."""
    info = get_image_resolution_info(img)
    mp = info["megapixels"]

    points = [(0.25, 0.08), (1.0, 0.12), (4.0, 0.15)]
    if mp <= points[0][0]:
        denoise = points[0][1]
    elif mp >= points[-1][0]:
        denoise = points[-1][1]
    else:
        denoise = points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= mp <= x1:
                ratio = (mp - x0) / (x1 - x0)
                denoise = y0 + ratio * (y1 - y0)
                break

    denoise *= scale
    denoise = max(min_val, min(max_val, denoise))
    return round(denoise, 3)
