from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PatchParams:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle_deg: float
    opacity: float
    intensity: float
    color_b: float
    color_g: float
    color_r: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass
class PatchSet:
    patches: list[PatchParams]

    def to_dict(self) -> dict[str, list[dict[str, float]]]:
        return {"patches": [patch.to_dict() for patch in self.patches]}


def _clip_params(params: PatchParams, height: int, width: int) -> PatchParams:
    min_radius = max(4.0, min(height, width) * 0.015)
    return PatchParams(
        center_x=float(np.clip(params.center_x, 0, width - 1)),
        center_y=float(np.clip(params.center_y, 0, height - 1)),
        radius_x=float(np.clip(params.radius_x, min_radius, width * 0.55)),
        radius_y=float(np.clip(params.radius_y, min_radius, height * 0.55)),
        angle_deg=float(params.angle_deg % 180),
        opacity=float(np.clip(params.opacity, 0.03, 0.85)),
        intensity=float(np.clip(params.intensity, 0.05, 1.8)),
        color_b=float(np.clip(params.color_b, 0, 255)),
        color_g=float(np.clip(params.color_g, 0, 255)),
        color_r=float(np.clip(params.color_r, 0, 255)),
    )


def random_patch_params(
    rng: np.random.Generator,
    image_shape: tuple[int, int, int],
    region: tuple[float, float, float, float] | None = None,
    min_size_frac: float = 0.04,
    max_size_frac: float = 0.18,
) -> PatchParams:
    height, width = image_shape[:2]
    if region:
        x1, y1, x2, y2 = region
        center_x = rng.uniform(x1, x2)
        center_y = rng.uniform(y1, y2)
        base = max(x2 - x1, y2 - y1, 1.0)
    else:
        center_x = rng.uniform(0, width - 1)
        center_y = rng.uniform(0, height - 1)
        base = min(height, width)

    min_radius = max(4.0, base * min_size_frac)
    max_radius = max(min_radius + 1.0, base * max_size_frac)
    warm = rng.uniform(0.78, 1.0)
    blue = rng.uniform(0.45, 0.82)

    return _clip_params(
        PatchParams(
            center_x=center_x,
            center_y=center_y,
            radius_x=rng.uniform(min_radius, max_radius),
            radius_y=rng.uniform(min_radius, max_radius),
            angle_deg=rng.uniform(0, 180),
            opacity=rng.uniform(0.18, 0.68),
            intensity=rng.uniform(0.25, 1.45),
            color_b=255 * blue,
            color_g=255 * warm,
            color_r=255,
        ),
        height,
        width,
    )


def mutate_patch_params(
    rng: np.random.Generator,
    params: PatchParams,
    image_shape: tuple[int, int, int],
    scale: float,
    region: tuple[float, float, float, float] | None = None,
) -> PatchParams:
    height, width = image_shape[:2]
    if region:
        x1, y1, x2, y2 = region
        base = max(x2 - x1, y2 - y1, 1.0)
    else:
        base = min(height, width)
    mutated = PatchParams(
        center_x=params.center_x + rng.normal(0, base * 0.10 * scale),
        center_y=params.center_y + rng.normal(0, base * 0.10 * scale),
        radius_x=params.radius_x * np.exp(rng.normal(0, 0.45 * scale)),
        radius_y=params.radius_y * np.exp(rng.normal(0, 0.45 * scale)),
        angle_deg=params.angle_deg + rng.normal(0, 45 * scale),
        opacity=params.opacity + rng.normal(0, 0.12 * scale),
        intensity=params.intensity + rng.normal(0, 0.25 * scale),
        color_b=params.color_b + rng.normal(0, 35 * scale),
        color_g=params.color_g + rng.normal(0, 25 * scale),
        color_r=params.color_r + rng.normal(0, 15 * scale),
    )
    return _clip_params(mutated, height, width)


def random_patch_set(
    rng: np.random.Generator,
    image_shape: tuple[int, int, int],
    region: tuple[float, float, float, float] | None = None,
    glare_count: int = 4,
    min_size_frac: float = 0.025,
    max_size_frac: float = 0.18,
) -> PatchSet:
    return PatchSet(
        [
            random_patch_params(
                rng,
                image_shape,
                region=region,
                min_size_frac=min_size_frac,
                max_size_frac=max_size_frac,
            )
            for _ in range(max(1, glare_count))
        ]
    )


def mutate_patch_set(
    rng: np.random.Generator,
    patch_set: PatchSet,
    image_shape: tuple[int, int, int],
    scale: float,
    region: tuple[float, float, float, float] | None = None,
    min_size_frac: float = 0.025,
    max_size_frac: float = 0.18,
) -> PatchSet:
    patches = []
    for patch in patch_set.patches:
        if rng.random() < 0.18 * scale:
            patches.append(
                random_patch_params(
                    rng,
                    image_shape,
                    region=region,
                    min_size_frac=min_size_frac,
                    max_size_frac=max_size_frac,
                )
            )
        else:
            patches.append(mutate_patch_params(rng, patch, image_shape, scale=scale, region=region))
    return PatchSet(patches)


def render_patch_mask(image_shape: tuple[int, int, int], params: PatchParams) -> np.ndarray:
    height, width = image_shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(params.angle_deg)
    dx = xx - params.center_x
    dy = yy - params.center_y
    xr = np.cos(theta) * dx + np.sin(theta) * dy
    yr = -np.sin(theta) * dx + np.cos(theta) * dy
    distance = (xr / max(params.radius_x, 1e-6)) ** 2 + (yr / max(params.radius_y, 1e-6)) ** 2
    mask = np.exp(-3.2 * distance).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=0.9, sigmaY=0.9)


def render_light_mask(
    image_shape: tuple[int, int, int],
    params: PatchParams | PatchSet,
    clip_region: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    if isinstance(params, PatchSet):
        mask = np.zeros(image_shape[:2], dtype=np.float32)
        for patch in params.patches:
            patch_mask = render_patch_mask(image_shape, patch) * patch.opacity
            mask = np.maximum(mask, patch_mask)
    else:
        mask = render_patch_mask(image_shape, params) * params.opacity
    return clip_mask_to_region(mask, clip_region)


def clip_mask_to_region(mask: np.ndarray, region: tuple[float, float, float, float] | None) -> np.ndarray:
    if region is None:
        return mask
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = region
    x1 = int(np.clip(np.floor(x1), 0, width))
    y1 = int(np.clip(np.floor(y1), 0, height))
    x2 = int(np.clip(np.ceil(x2), 0, width))
    y2 = int(np.clip(np.ceil(y2), 0, height))
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


def apply_light_patch(
    image_bgr: np.ndarray,
    params: PatchParams | PatchSet,
    clip_region: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(params, PatchSet):
        out = image_bgr.copy()
        combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        for patch in params.patches:
            out, mask = apply_light_patch(out, patch, clip_region=clip_region)
            combined_mask = np.maximum(combined_mask, mask)
        return out, combined_mask

    params = _clip_params(params, *image_bgr.shape[:2])
    image = image_bgr.astype(np.float32) / 255.0
    mask = clip_mask_to_region(render_patch_mask(image_bgr.shape, params), clip_region) * params.opacity
    tint = np.array([params.color_b, params.color_g, params.color_r], dtype=np.float32) / 255.0

    highlight = tint.reshape(1, 1, 3) * params.intensity * 0.48
    glare = mask[:, :, None]
    out = image * (1.0 + 0.42 * glare) + highlight * glare
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8), mask
