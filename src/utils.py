from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


DEFAULT_CONFIG: dict[str, Any] = {
    "margin_percent": 1.0,
    "debug": True,
    "color_correction_strength": 0.6,
    "keep_white_border": False,
    "accept_already_cropped": False,
    "rotate_to_landscape": True,
    "output_quality": 95,
    "gamma": 1.0,
    "saturation": 1.05,
    "restoration": {
        "enabled": True,
        "glare_reduction_strength": 0.35,
        "white_balance_strength": 0.45,
        "age_cast_reduction_strength": 0.55,
        "local_contrast_strength": 0.45,
        "shadow_recovery_strength": 0.28,
        "vibrance": 1.12,
    },
    "gemini": {
        "enabled": False,
        "mode": "fallback",
        "model": "gemini-3.5-flash",
        "min_confidence": 0.55,
        "max_retries": 2,
        "retry_delay_seconds": 2.0,
        "request_timeout_seconds": 25.0,
        "orientation_check": True,
        "orientation_min_confidence": 0.65,
        "orientation_max_retries": 1,
        "fallback_methods": ["album_edges", "foreground_bbox", "full_frame", "not_found", "unreasonable"],
    },
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            user_config = json.load(file)
        config.update(user_config)
    return config


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "cropped": output_dir / "cropped",
        "debug": output_dir / "debug",
        "needs_review": output_dir / "needs_review",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def iter_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path: Path, image_bgr: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    save_kwargs: dict[str, Any] = {}
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": quality, "optimize": True})
    elif path.suffix.lower() == ".webp":
        save_kwargs.update({"quality": quality, "method": 6})
    image.save(path, **save_kwargs)


def safe_output_name(source_path: Path, extension: str = ".jpg") -> str:
    return f"{source_path.stem}{extension}"


def copy_to_needs_review(source_path: Path, review_dir: Path) -> Path:
    review_path = review_dir / source_path.name
    counter = 1
    while review_path.exists():
        review_path = review_dir / f"{source_path.stem}_{counter}{source_path.suffix}"
        counter += 1
    shutil.copy2(source_path, review_path)
    return review_path


def resize_for_processing(image: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_side:
        return image.copy(), 1.0
    scale = max_side / largest
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
