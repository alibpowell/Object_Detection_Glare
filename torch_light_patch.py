from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ExportedGlints:
    center: list[list[float]]
    radius: list[list[float]]
    angle: list[float]
    opacity: list[float]
    intensity: list[float]
    color_rgb: list[list[float]]


def _meshgrid(height: int, width: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return xx, yy


def _box_clip_mask(height: int, width: int, box_xyxy: torch.Tensor, device) -> torch.Tensor:
    xx, yy = _meshgrid(height, width, device)
    x1, y1, x2, y2 = box_xyxy
    return ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()


class GradientLightPatch(nn.Module):
    def __init__(
        self,
        glare_count: int = 5,
        min_size_frac: float = 0.025,
        max_size_frac: float = 0.18,
        seed: int = 0,
        device: str = "cuda:0",
    ):
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.glare_count = glare_count
        self.min_size_frac = min_size_frac
        self.max_size_frac = max_size_frac

        def randn(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float32)

        self.raw_center = nn.Parameter(randn(glare_count, 2))
        self.raw_radius = nn.Parameter(randn(glare_count, 2))
        self.raw_angle = nn.Parameter(randn(glare_count))
        self.raw_opacity = nn.Parameter(randn(glare_count))
        self.raw_intensity = nn.Parameter(randn(glare_count))
        self.raw_green = nn.Parameter(randn(glare_count))
        self.raw_blue = nn.Parameter(randn(glare_count))
        self.to(device)

    def add_glint(self, count: int = 1) -> None:
        if count <= 0:
            return
        device = self.raw_center.device

        def expand(parameter: nn.Parameter, shape: tuple[int, ...]) -> nn.Parameter:
            extra = torch.randn(*shape, device=device, dtype=parameter.dtype)
            return nn.Parameter(torch.cat([parameter.detach(), extra], dim=0))

        self.raw_center = expand(self.raw_center, (count, 2))
        self.raw_radius = expand(self.raw_radius, (count, 2))
        self.raw_angle = expand(self.raw_angle, (count,))
        self.raw_opacity = expand(self.raw_opacity, (count,))
        self.raw_intensity = expand(self.raw_intensity, (count,))
        self.raw_green = expand(self.raw_green, (count,))
        self.raw_blue = expand(self.raw_blue, (count,))
        self.glare_count += count

    def randomize_glint(self, index: int) -> None:
        if index < 0 or index >= self.glare_count:
            raise IndexError(f"glint index {index} is outside 0..{self.glare_count - 1}")
        with torch.no_grad():
            self.raw_center[index].normal_()
            self.raw_radius[index].normal_()
            self.raw_angle[index].normal_()
            self.raw_opacity[index].normal_()
            self.raw_intensity[index].normal_()
            self.raw_green[index].normal_()
            self.raw_blue[index].normal_()

    def weakest_glint_index(self, box_xyxy: torch.Tensor) -> int:
        with torch.no_grad():
            _, _, radius_x, radius_y, _, opacity, intensity, _ = self.bounded_params(box_xyxy)
            contribution = radius_x * radius_y * opacity * intensity
            return int(torch.argmin(contribution).item())

    def bounded_params(self, box_xyxy: torch.Tensor):
        x1, y1, x2, y2 = box_xyxy
        box_w = torch.clamp(x2 - x1, min=1.0)
        box_h = torch.clamp(y2 - y1, min=1.0)
        base = torch.maximum(box_w, box_h)

        center_rel = torch.sigmoid(self.raw_center)
        center_x = x1 + center_rel[:, 0] * box_w
        center_y = y1 + center_rel[:, 1] * box_h

        radius_rel = self.min_size_frac + torch.sigmoid(self.raw_radius) * (
            self.max_size_frac - self.min_size_frac
        )
        radius_x = radius_rel[:, 0] * base
        radius_y = radius_rel[:, 1] * base

        angle = torch.sigmoid(self.raw_angle) * torch.pi
        opacity = 0.02 + torch.sigmoid(self.raw_opacity) * 0.38
        intensity = 0.05 + torch.sigmoid(self.raw_intensity) * 0.75
        red = torch.ones_like(opacity)
        green = 0.74 + torch.sigmoid(self.raw_green) * 0.24
        blue = 0.45 + torch.sigmoid(self.raw_blue) * 0.35
        color = torch.stack([red, green, blue], dim=1)

        return center_x, center_y, radius_x, radius_y, angle, opacity, intensity, color

    def forward(self, image_rgb: torch.Tensor, box_xyxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = image_rgb.shape
        device = image_rgb.device
        xx, yy = _meshgrid(height, width, device)
        clip = _box_clip_mask(height, width, box_xyxy, device)
        out = image_rgb
        combined_mask = torch.zeros((height, width), device=device, dtype=torch.float32)

        center_x, center_y, radius_x, radius_y, angle, opacity, intensity, color = self.bounded_params(box_xyxy)
        for i in range(self.glare_count):
            dx = xx - center_x[i]
            dy = yy - center_y[i]
            cos_a = torch.cos(angle[i])
            sin_a = torch.sin(angle[i])
            xr = cos_a * dx + sin_a * dy
            yr = -sin_a * dx + cos_a * dy
            dist = (xr / torch.clamp(radius_x[i], min=1e-4)) ** 2 + (
                yr / torch.clamp(radius_y[i], min=1e-4)
            ) ** 2
            mask = torch.exp(-3.2 * dist) * opacity[i] * clip
            glare = mask.view(1, 1, height, width)
            tint = color[i].view(1, 3, 1, 1)
            out = out * (1.0 + 0.22 * glare) + tint * intensity[i] * 0.28 * glare
            out = torch.clamp(out, 0.0, 1.0)
            combined_mask = torch.maximum(combined_mask, mask)

        return out, combined_mask

    def naturalness_loss(self, mask: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = box_xyxy
        box_area = torch.clamp((x2 - x1) * (y2 - y1), min=1.0)
        area = mask.sum() / box_area
        peak = mask.max()
        tv_y = torch.mean(torch.abs(mask[1:, :] - mask[:-1, :]))
        tv_x = torch.mean(torch.abs(mask[:, 1:] - mask[:, :-1]))
        return 2.5 * area + 0.60 * peak + 0.08 * (tv_x + tv_y)

    def export(self, box_xyxy: torch.Tensor) -> ExportedGlints:
        with torch.no_grad():
            center_x, center_y, radius_x, radius_y, angle, opacity, intensity, color = self.bounded_params(box_xyxy)
            x1, y1, x2, y2 = box_xyxy
            box_w = torch.clamp(x2 - x1, min=1.0)
            box_h = torch.clamp(y2 - y1, min=1.0)
            base = torch.maximum(box_w, box_h)
            center_rel = torch.stack([(center_x - x1) / box_w, (center_y - y1) / box_h], dim=1)
            radius_rel = torch.stack([radius_x / base, radius_y / base], dim=1)
            return ExportedGlints(
                center=center_rel.detach().cpu().tolist(),
                radius=radius_rel.detach().cpu().tolist(),
                angle=angle.detach().cpu().tolist(),
                opacity=opacity.detach().cpu().tolist(),
                intensity=intensity.detach().cpu().tolist(),
                color_rgb=color.detach().cpu().tolist(),
            )


def render_exported_glints(
    image_rgb: torch.Tensor,
    box_xyxy: torch.Tensor,
    glints: ExportedGlints,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = image_rgb.shape
    device = image_rgb.device
    xx, yy = _meshgrid(height, width, device)
    clip = _box_clip_mask(height, width, box_xyxy, device)
    x1, y1, x2, y2 = box_xyxy
    box_w = torch.clamp(x2 - x1, min=1.0)
    box_h = torch.clamp(y2 - y1, min=1.0)
    base = torch.maximum(box_w, box_h)

    out = image_rgb
    combined_mask = torch.zeros((height, width), device=device, dtype=torch.float32)
    for center_rel, radius_rel, angle, opacity, intensity, color_rgb in zip(
        glints.center,
        glints.radius,
        glints.angle,
        glints.opacity,
        glints.intensity,
        glints.color_rgb,
    ):
        cx = x1 + float(center_rel[0]) * box_w
        cy = y1 + float(center_rel[1]) * box_h
        rx = max(float(radius_rel[0]), 1e-4) * base
        ry = max(float(radius_rel[1]), 1e-4) * base
        angle_t = torch.tensor(float(angle), device=device)
        dx = xx - cx
        dy = yy - cy
        xr = torch.cos(angle_t) * dx + torch.sin(angle_t) * dy
        yr = -torch.sin(angle_t) * dx + torch.cos(angle_t) * dy
        dist = (xr / rx) ** 2 + (yr / ry) ** 2
        mask = torch.exp(-3.2 * dist) * float(opacity) * clip
        glare = mask.view(1, 1, height, width)
        tint = torch.tensor(color_rgb, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        out = out * (1.0 + 0.22 * glare) + tint * float(intensity) * 0.28 * glare
        out = torch.clamp(out, 0.0, 1.0)
        combined_mask = torch.maximum(combined_mask, mask)
    return out, combined_mask
