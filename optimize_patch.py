from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from light_patch import (
    PatchParams,
    PatchSet,
    apply_light_patch,
    mutate_patch_params,
    mutate_patch_set,
    random_patch_params,
    random_patch_set,
)
from yolo_utils import detections_from_result, max_detection_score, run_yolo, run_yolo_batch


@dataclass
class AttackConfig:
    mode: str
    source_class_id: int | None
    target_class_id: int | None
    region: tuple[float, float, float, float] | None
    source_targets: list[dict] | None
    iterations: int
    seed: int
    conf: float
    iou: float
    imgsz: int
    device: str
    clip_patch_to_region: bool = False
    candidates_per_iter: int = 6
    glare_count: int = 4
    naturalness_weight: float = 0.05
    min_size_frac: float = 0.025
    max_size_frac: float = 0.18


def source_targets_score(detections: list[dict], targets: list[dict]) -> float:
    if not targets:
        return 0.0
    scores = [
        max_detection_score(detections, target["class_id"], tuple(target["xyxy"]))
        for target in targets
    ]
    return float(np.mean(scores))


def _score_candidate(
    model,
    image_bgr,
    params: PatchParams | PatchSet,
    config: AttackConfig,
) -> tuple[float, np.ndarray, list[dict]]:
    clip_region = config.region if config.clip_patch_to_region else None
    attacked, mask = apply_light_patch(image_bgr, params, clip_region=clip_region)
    result = run_yolo(model, attacked, device=config.device, conf=config.conf, iou=config.iou, imgsz=config.imgsz)
    detections = detections_from_result(result)

    if config.mode == "disappear":
        if config.source_targets:
            model_score = source_targets_score(detections, config.source_targets)
        elif config.source_class_id is not None:
            model_score = max_detection_score(detections, config.source_class_id, config.region)
        else:
            raise ValueError("source_class_id is required for disappear mode")
    elif config.mode == "targeted":
        if config.target_class_id is None:
            raise ValueError("target_class_id is required for targeted mode")
        model_score = max_detection_score(detections, config.target_class_id, config.region)
    else:
        raise ValueError(f"Unknown attack mode: {config.mode}")

    if config.mode == "disappear":
        visibility_penalty = config.naturalness_weight * (float(mask.mean()) + 0.15 * float(mask.max()))
        model_score += visibility_penalty

    return model_score, attacked, detections


def _score_detections(detections: list[dict], mask: np.ndarray, config: AttackConfig) -> float:
    if config.mode == "disappear":
        if config.source_targets:
            model_score = source_targets_score(detections, config.source_targets)
        elif config.source_class_id is not None:
            model_score = max_detection_score(detections, config.source_class_id, config.region)
        else:
            raise ValueError("source_class_id is required for disappear mode")
        visibility_penalty = config.naturalness_weight * (float(mask.mean()) + 0.15 * float(mask.max()))
        return model_score + visibility_penalty

    if config.mode == "targeted":
        if config.target_class_id is None:
            raise ValueError("target_class_id is required for targeted mode")
        return max_detection_score(detections, config.target_class_id, config.region)

    raise ValueError(f"Unknown attack mode: {config.mode}")


def is_better(mode: str, candidate_score: float, best_score: float) -> bool:
    if mode == "disappear":
        return candidate_score < best_score
    if mode == "targeted":
        return candidate_score > best_score
    raise ValueError(f"Unknown attack mode: {mode}")


def _make_candidate_params(rng, best_params, image_shape, iteration: int, config: AttackConfig):
    if iteration < max(8, config.iterations // 5):
        if config.glare_count > 1:
            return random_patch_set(
                rng,
                image_shape,
                region=config.region,
                glare_count=config.glare_count,
                min_size_frac=config.min_size_frac,
                max_size_frac=config.max_size_frac,
            )
        return random_patch_params(
            rng,
            image_shape,
            region=config.region,
            min_size_frac=config.min_size_frac,
            max_size_frac=config.max_size_frac,
        )

    scale = max(0.05, 1.0 - iteration / max(config.iterations - 1, 1))
    if isinstance(best_params, PatchSet):
        return mutate_patch_set(
            rng,
            best_params,
            image_shape,
            scale=scale,
            region=config.region,
            min_size_frac=config.min_size_frac,
            max_size_frac=config.max_size_frac,
        )
    return mutate_patch_params(
        rng,
        best_params,
        image_shape,
        scale=scale,
        region=config.region,
    )


def optimize_light_patch(model, image_bgr, config: AttackConfig):
    rng = np.random.default_rng(config.seed)
    progress = []

    if config.glare_count > 1:
        best_params = random_patch_set(
            rng,
            image_bgr.shape,
            region=config.region,
            glare_count=config.glare_count,
            min_size_frac=config.min_size_frac,
            max_size_frac=config.max_size_frac,
        )
    else:
        best_params = random_patch_params(
            rng,
            image_bgr.shape,
            region=config.region,
            min_size_frac=config.min_size_frac,
            max_size_frac=config.max_size_frac,
        )
    best_score, best_image, best_detections = _score_candidate(model, image_bgr, best_params, config)

    for iteration in range(config.iterations):
        candidate_scores = []
        candidate_params = []
        candidate_images = []
        candidate_masks = []
        clip_region = config.region if config.clip_patch_to_region else None

        for _ in range(max(1, config.candidates_per_iter)):
            params = _make_candidate_params(rng, best_params, image_bgr.shape, iteration, config)
            attacked, mask = apply_light_patch(image_bgr, params, clip_region=clip_region)
            candidate_params.append(params)
            candidate_images.append(attacked)
            candidate_masks.append(mask)

        results = run_yolo_batch(
            model,
            candidate_images,
            device=config.device,
            conf=config.conf,
            iou=config.iou,
            imgsz=config.imgsz,
        )

        last_score = None
        for params, attacked, mask, result in zip(candidate_params, candidate_images, candidate_masks, results):
            detections = detections_from_result(result)
            score = _score_detections(detections, mask, config)
            last_score = score
            candidate_scores.append(score)
            if is_better(config.mode, score, best_score):
                best_score = score
                best_params = params
                best_image = attacked
                best_detections = detections

        progress.append(
            {
                "iteration": iteration,
                "candidate_score": float(min(candidate_scores) if config.mode == "disappear" else max(candidate_scores)),
                "best_score": float(best_score),
                "patch": best_params.to_dict(),
            }
        )

        if iteration % 10 == 0 or iteration == config.iterations - 1:
            print(f"[{iteration:04d}/{config.iterations}] candidate={last_score:.4f} best={best_score:.4f}")

    return {
        "best_score": float(best_score),
        "best_params": best_params,
        "best_image": best_image,
        "best_detections": best_detections,
        "progress": progress,
    }
