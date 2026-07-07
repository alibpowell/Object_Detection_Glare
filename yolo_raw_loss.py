from __future__ import annotations

import torch


def raw_predictions(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def box_iou_xyxy(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    ix1 = torch.maximum(boxes[:, 0], box[0])
    iy1 = torch.maximum(boxes[:, 1], box[1])
    ix2 = torch.minimum(boxes[:, 2], box[2])
    iy2 = torch.minimum(boxes[:, 3], box[3])
    inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
    area_a = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0)
    area_b = torch.clamp(box[2] - box[0], min=0) * torch.clamp(box[3] - box[1], min=0)
    return inter / torch.clamp(area_a + area_b - inter, min=1e-6)


def disappearance_loss(
    model_output,
    source_class_id: int,
    target_box_xyxy: torch.Tensor,
    min_iou: float = 0.03,
    temperature: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    preds = raw_predictions(model_output)
    if preds.ndim != 3:
        raise ValueError(f"Expected YOLO raw prediction shape [B, C, N], got {tuple(preds.shape)}")
    pred = preds[0].transpose(0, 1)
    boxes = xywh_to_xyxy(pred[:, :4])
    class_scores = pred[:, 4 + source_class_id]

    centers = pred[:, :2]
    inside = (
        (centers[:, 0] >= target_box_xyxy[0])
        & (centers[:, 0] <= target_box_xyxy[2])
        & (centers[:, 1] >= target_box_xyxy[1])
        & (centers[:, 1] <= target_box_xyxy[3])
    )
    overlaps = box_iou_xyxy(boxes, target_box_xyxy) > min_iou
    relevant = inside | overlaps

    if int(relevant.sum().item()) == 0:
        relevant = torch.ones_like(class_scores, dtype=torch.bool)

    selected = class_scores[relevant]
    smooth_max = temperature * torch.logsumexp(selected / temperature, dim=0)
    observed_max = selected.max().detach()
    return smooth_max, observed_max, int(relevant.sum().item())

