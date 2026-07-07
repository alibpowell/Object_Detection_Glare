from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import pillow_avif  # noqa: F401
except Exception:
    pillow_avif = None


def load_image_bgr(path: str | Path) -> np.ndarray:
    """Load an image as uint8 BGR, with a Pillow fallback for formats like AVIF."""
    path = Path(path)
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    with Image.open(path) as pil_image:
        rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image_bgr)
    if not ok:
        raise ValueError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
