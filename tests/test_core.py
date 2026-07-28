from __future__ import annotations

import numpy as np
import pytest
import torch

from image_io import load_image_bgr, save_image
from light_patch import PatchParams, apply_light_patch, render_light_mask
from torch_light_patch import ExportedGlints, render_exported_glints
from yolo_raw_loss import disappearance_loss
from yolo_utils import box_iou, max_detection_score


def test_box_iou_handles_overlap_separation_and_degenerate_boxes() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert box_iou((0, 0, 2, 2), (3, 3, 4, 4)) == 0.0
    assert box_iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_max_detection_score_filters_by_class_and_region() -> None:
    detections = [
        {"class_id": 0, "confidence": 0.45, "xyxy": [0, 0, 10, 10]},
        {"class_id": 0, "confidence": 0.80, "xyxy": [20, 20, 30, 30]},
        {"class_id": 1, "confidence": 0.95, "xyxy": [0, 0, 10, 10]},
    ]

    assert max_detection_score(detections, 0) == pytest.approx(0.80)
    assert max_detection_score(detections, 0, (0, 0, 12, 12)) == pytest.approx(0.45)
    assert max_detection_score(detections, 2) == 0.0


def test_disappearance_loss_is_class_agnostic_and_differentiable() -> None:
    # YOLO-like shape: [batch, x/y/w/h + two classes, predictions].
    predictions = torch.tensor(
        [[[5.0], [5.0], [4.0], [4.0], [0.20], [0.90]]],
        requires_grad=True,
    )
    target = torch.tensor([2.0, 2.0, 8.0, 8.0])

    loss, observed_max, relevant_count = disappearance_loss(
        predictions,
        source_class_id=0,
        target_box_xyxy=target,
        temperature=0.05,
    )

    assert observed_max.item() == pytest.approx(0.90)
    assert relevant_count == 1
    loss.backward()
    assert predictions.grad is not None
    assert predictions.grad[0, 5, 0] > 0


def test_numpy_light_mask_is_clipped_to_the_requested_region() -> None:
    params = PatchParams(8, 8, 5, 5, 0, 0.5, 1.0, 180, 220, 255)
    mask = render_light_mask((16, 16, 3), params, clip_region=(4, 5, 12, 13))

    assert mask.shape == (16, 16)
    assert mask[5:13, 4:12].max() > 0
    outside = mask.copy()
    outside[5:13, 4:12] = 0
    assert np.count_nonzero(outside) == 0


def test_apply_light_patch_preserves_shape_type_and_range() -> None:
    image = np.full((12, 14, 3), 80, dtype=np.uint8)
    params = PatchParams(7, 6, 4, 3, 20, 0.5, 1.0, 180, 220, 255)

    attacked, mask = apply_light_patch(image, params)

    assert attacked.shape == image.shape
    assert attacked.dtype == np.uint8
    assert mask.shape == image.shape[:2]
    assert attacked.min() >= 0 and attacked.max() <= 255
    assert not np.array_equal(attacked, image)


def test_exported_glint_renderer_changes_only_inside_target_box() -> None:
    image = torch.zeros((1, 3, 12, 12), dtype=torch.float32)
    box = torch.tensor([3.0, 2.0, 9.0, 10.0])
    glints = ExportedGlints(
        center=[[0.5, 0.5]],
        radius=[[0.25, 0.25]],
        angle=[0.0],
        opacity=[0.7],
        intensity=[1.0],
        color_rgb=[[1.0, 0.9, 0.6]],
    )

    rendered, mask = render_exported_glints(image, box, glints)

    assert rendered.shape == image.shape
    assert mask[2:11, 3:10].max() > 0
    outside = mask.clone()
    outside[2:11, 3:10] = 0
    assert torch.count_nonzero(outside) == 0
    assert torch.all((rendered >= 0) & (rendered <= 1))


def test_image_io_round_trip(tmp_path) -> None:
    original = np.zeros((7, 9, 3), dtype=np.uint8)
    original[:, :, 1] = 127
    path = tmp_path / "round_trip.png"

    save_image(path, original)
    loaded = load_image_bgr(path)

    assert np.array_equal(loaded, original)
