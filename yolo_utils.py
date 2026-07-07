from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics import YOLO


def load_yolo(weights: str, device: str) -> YOLO:
    model = YOLO(weights)
    model.to(device)
    return model


def run_yolo(
    model: YOLO,
    image_bgr: np.ndarray,
    device: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
):
    return model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]


def run_yolo_batch(
    model: YOLO,
    images_bgr: list[np.ndarray],
    device: str,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
):
    if not images_bgr:
        return []
    return model.predict(
        source=images_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )


def names_dict(model: YOLO) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {idx: str(name) for idx, name in enumerate(names)}


def class_id_from_name(model: YOLO, class_value: str | int) -> int:
    if isinstance(class_value, int):
        return class_value
    value = str(class_value).strip()
    if value.isdigit():
        return int(value)

    names = names_dict(model)
    lowered = value.lower()
    exact = [idx for idx, name in names.items() if name.lower() == lowered]
    if exact:
        return exact[0]

    partial = [idx for idx, name in names.items() if lowered in name.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        matches = ", ".join(f"{idx}:{names[idx]}" for idx in partial[:8])
        raise ValueError(f"Class name {value!r} is ambiguous. Matches: {matches}")
    raise ValueError(f"Class {value!r} was not found in YOLO model names")


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names
    return [
        {
            "xyxy": [float(v) for v in box],
            "confidence": float(conf),
            "class_id": int(cls),
            "class_name": str(names[int(cls)]),
        }
        for box, conf, cls in zip(boxes, confs, classes)
    ]


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_detection_score(
    detections: list[dict[str, Any]],
    class_id: int,
    region: tuple[float, float, float, float] | None = None,
    min_iou: float = 0.05,
) -> float:
    scores = []
    for det in detections:
        if det["class_id"] != class_id:
            continue
        if region is not None and box_iou(tuple(det["xyxy"]), region) < min_iou:
            continue
        scores.append(det["confidence"])
    return max(scores) if scores else 0.0


def top_detection(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not detections:
        return None
    return max(detections, key=lambda det: det["confidence"])
