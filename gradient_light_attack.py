from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "object_detection_glare_matplotlib"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from image_io import ensure_dir, load_image_bgr, save_image
from torch_light_patch import GradientLightPatch, render_exported_glints
from yolo_raw_loss import disappearance_loss
from yolo_utils import (
    box_iou,
    class_id_from_name,
    detections_from_result,
    run_yolo,
)


def parse_region(value: str | None):
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region must be x1,y1,x2,y2")
    return tuple(parts)


def resolve_image_path(path: Path) -> Path:
    if path.exists():
        return path
    if not path.is_absolute() and path.parent == Path("."):
        input_path = Path("inputs") / path.name
        if input_path.exists():
            print(f"Image {path} was not found; using {input_path}.")
            return input_path
    return path


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        index = int(device.split(":")[1]) if ":" in device else 0
        print(f"CUDA device: {torch.cuda.get_device_name(index)}")
        print(f"Torch CUDA: {torch.version.cuda}")
    elif device == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        print("Apple GPU device: MPS")
    else:
        print("Running on CPU.")


def image_bgr_to_tensor_rgb(image_bgr: np.ndarray, device: str, size: int | None = None) -> torch.Tensor:
    if size is not None:
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32) / 255.0
    return tensor.permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_rgb_to_bgr(image_rgb: torch.Tensor) -> np.ndarray:
    arr = image_rgb.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    rgb = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def scale_box_to_square(box, width: int, height: int, size: int, device: str) -> torch.Tensor:
    x1, y1, x2, y2 = box
    return torch.tensor(
        [x1 * size / width, y1 * size / height, x2 * size / width, y2 * size / height],
        device=device,
        dtype=torch.float32,
    )


def box_tensor(box, device: str) -> torch.Tensor:
    return torch.tensor(box, device=device, dtype=torch.float32)


def source_targets_from_detections(detections, source_class_id, region):
    targets = []
    for det in detections:
        if source_class_id is not None and det["class_id"] != source_class_id:
            continue
        if region is not None and box_iou(tuple(det["xyxy"]), region) < 0.05:
            continue
        targets.append(
            {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "xyxy": det["xyxy"],
                "original_confidence": det["confidence"],
            }
        )
    return targets


def max_any_detection_score(
    detections: list[dict],
    region: tuple[float, float, float, float],
    min_iou: float = 0.05,
) -> tuple[float, dict | None]:
    best_score = 0.0
    best_detection = None
    for det in detections:
        if box_iou(tuple(det["xyxy"]), region) < min_iou:
            continue
        if det["confidence"] > best_score:
            best_score = det["confidence"]
            best_detection = det
    return best_score, best_detection


def glint_count(glints) -> int:
    return len(glints.center)


def remove_glint(glints, index: int):
    from torch_light_patch import ExportedGlints

    return ExportedGlints(
        center=[value for idx, value in enumerate(glints.center) if idx != index],
        radius=[value for idx, value in enumerate(glints.radius) if idx != index],
        angle=[value for idx, value in enumerate(glints.angle) if idx != index],
        opacity=[value for idx, value in enumerate(glints.opacity) if idx != index],
        intensity=[value for idx, value in enumerate(glints.intensity) if idx != index],
        color_rgb=[value for idx, value in enumerate(glints.color_rgb) if idx != index],
    )


def evaluate_glints_on_original(
    detect_model,
    base_image_original: torch.Tensor,
    target_box_original: torch.Tensor,
    glints,
    args,
    device: str,
) -> tuple[float, dict | None, torch.Tensor]:
    attacked_original, _ = render_exported_glints(base_image_original, target_box_original, glints)
    result = run_yolo(
        detect_model,
        tensor_rgb_to_bgr(attacked_original),
        device=device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
    detections = detections_from_result(result)
    target_box = tuple(float(v) for v in target_box_original.detach().cpu().tolist())
    score, detection = max_any_detection_score(detections, target_box)
    return score, detection, attacked_original


def prune_successful_glints(
    detect_model,
    base_image_original: torch.Tensor,
    target_box_original: torch.Tensor,
    glints,
    target: dict,
    args,
    device: str,
) -> tuple[object, list[dict], torch.Tensor]:
    events = []
    current_glints = glints
    current_score, current_detection, current_image = evaluate_glints_on_original(
        detect_model,
        base_image_original,
        target_box_original,
        current_glints,
        args,
        device,
    )
    if current_score != 0.0:
        return current_glints, events, current_image

    changed = True
    while changed and glint_count(current_glints) > 1:
        changed = False
        for index in range(glint_count(current_glints)):
            candidate_glints = remove_glint(current_glints, index)
            score, detection, image = evaluate_glints_on_original(
                detect_model,
                base_image_original,
                target_box_original,
                candidate_glints,
                args,
                device,
            )
            if score == 0.0:
                events.append(
                    {
                        "removed_glint_index": index,
                        "glare_count_before": glint_count(current_glints),
                        "glare_count_after": glint_count(candidate_glints),
                        "post_prune_score": score,
                        "post_prune_class": detection["class_name"] if detection else None,
                    }
                )
                current_glints = candidate_glints
                current_image = image
                changed = True
                break

    return current_glints, events, current_image


def apply_auto_attack_defaults(args) -> None:
    if not args.auto_attack:
        return
    args.until_disappeared = True
    if args.max_steps <= 0:
        args.max_steps = 3500
    args.check_every = min(args.check_every, 25)
    args.glare_count = max(args.glare_count, 3)
    if args.max_glare_count > 0:
        args.max_glare_count = max(args.max_glare_count, 24)
    args.teleport_candidates = max(args.teleport_candidates, 12)
    args.teleport_steps = max(args.teleport_steps, 30)
    args.plateau_window = min(args.plateau_window, 60)
    args.growth_cooldown = min(args.growth_cooldown, 30)
    args.naturalness_weight = min(args.naturalness_weight, 0.03)
    args.max_size_frac = max(args.max_size_frac, 0.24)
    args.raw_topk = 64 if args.raw_topk <= 0 else min(args.raw_topk, 64)
    args.temperature = min(args.temperature, 0.03)
    args.auto_escalate = True


def write_progress(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def run_patch_step(
    raw_model,
    patch: GradientLightPatch,
    optimizer: torch.optim.Optimizer,
    base_image_square: torch.Tensor,
    target: dict,
    target_box_square: torch.Tensor,
    args,
):
    optimizer.zero_grad(set_to_none=True)
    state_before_step = clone_state(patch)
    attacked, mask = patch(base_image_square, target_box_square)
    output = raw_model(attacked)
    det_loss, raw_score, relevant_count = disappearance_loss(
        output,
        target["class_id"],
        target_box_square,
        min_iou=args.raw_iou,
        temperature=args.temperature,
        topk=args.raw_topk,
    )
    naturalness = patch.naturalness_loss(mask, target_box_square)
    loss = det_loss + args.naturalness_weight * naturalness
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "raw_score": float(raw_score.detach().item()),
        "naturalness": float(naturalness.detach().item()),
        "relevant_predictions": relevant_count,
        "attacked": attacked.detach(),
        "state": state_before_step,
    }


def export_from_state(
    patch: GradientLightPatch,
    state: dict[str, torch.Tensor],
    target_box_square: torch.Tensor,
) -> object:
    current_state = clone_state(patch)
    patch.load_state_dict(state)
    glints = patch.export(target_box_square)
    patch.load_state_dict(current_state)
    return glints


def try_teleport_weakest_glint(
    raw_model,
    patch: GradientLightPatch,
    base_image_square: torch.Tensor,
    target: dict,
    target_box_square: torch.Tensor,
    args,
    target_index: int,
    step: int,
    best_attack_score: float,
    best_loss: float,
) -> dict:
    original_state = clone_state(patch)
    weakest_index = patch.weakest_glint_index(target_box_square)
    best_trial = {
        "improved": False,
        "attack_score": best_attack_score,
        "loss": best_loss,
        "state": original_state,
        "image": None,
        "raw_score": None,
        "naturalness": None,
        "relevant_predictions": None,
        "glints": None,
        "candidate": None,
    }

    for candidate in range(args.teleport_candidates):
        patch.load_state_dict(original_state)
        patch.randomize_glint(weakest_index)
        trial_optimizer = torch.optim.Adam(patch.parameters(), lr=args.lr)
        trial_best_attack_score = float("inf")
        trial_best_loss = float("inf")
        trial_best_state = clone_state(patch)
        trial_best_image = None
        trial_best_raw = None
        trial_best_naturalness = None
        trial_best_relevant = None

        for _ in range(max(1, args.teleport_steps)):
            result = run_patch_step(
                raw_model,
                patch,
                trial_optimizer,
                base_image_square,
                target,
                target_box_square,
                args,
            )
            attack_score = result["raw_score"]
            loss_score = result["loss"]
            raw_improved = attack_score < trial_best_attack_score - args.plateau_delta
            raw_tied = abs(attack_score - trial_best_attack_score) <= args.plateau_delta
            if raw_improved or (raw_tied and loss_score < trial_best_loss):
                trial_best_attack_score = attack_score
                trial_best_loss = loss_score
                trial_best_state = result["state"]
                trial_best_image = result["attacked"]
                trial_best_raw = result["raw_score"]
                trial_best_naturalness = result["naturalness"]
                trial_best_relevant = result["relevant_predictions"]

        if trial_best_attack_score < best_trial["attack_score"] - args.teleport_delta:
            patch.load_state_dict(trial_best_state)
            best_trial = {
                "improved": True,
                "attack_score": trial_best_attack_score,
                "loss": trial_best_loss,
                "state": trial_best_state,
                "image": trial_best_image,
                "raw_score": trial_best_raw,
                "naturalness": trial_best_naturalness,
                "relevant_predictions": trial_best_relevant,
                "glints": patch.export(target_box_square),
                "candidate": candidate,
            }

    patch.load_state_dict(best_trial["state"] if best_trial["improved"] else original_state)
    return {
        "target_index": target_index,
        "step": step,
        "reason": "teleport",
        "glare_count": patch.glare_count,
        "teleported_glint": weakest_index,
        "teleport_candidates": args.teleport_candidates,
        "teleport_steps": args.teleport_steps,
        "best_attack_score_before": best_attack_score,
        "best_attack_score_after": best_trial["attack_score"],
        "best_loss_before": best_loss,
        "best_loss_after": best_trial["loss"],
        **best_trial,
    }


def auto_escalate_patch(
    patch: GradientLightPatch,
    args,
    step: int,
    target_index: int,
    actual_score: float,
    actual_detection: dict | None,
    escalation_count: int,
) -> tuple[bool, dict]:
    if not args.auto_escalate:
        return False, {}
    if args.auto_escalation_limit > 0 and escalation_count >= args.auto_escalation_limit:
        return False, {}

    before_count = patch.glare_count
    before_max_size = patch.max_size_frac
    before_naturalness = args.naturalness_weight
    before_lr = args.lr

    changed = False
    can_grow = args.max_glare_count <= 0 or patch.glare_count < args.max_glare_count
    if can_grow:
        if args.max_glare_count <= 0:
            add_count = args.auto_glints_per_escalation
        else:
            add_count = min(args.auto_glints_per_escalation, args.max_glare_count - patch.glare_count)
        patch.add_glint(add_count)
        changed = changed or add_count > 0

    if patch.max_size_frac < args.auto_max_size_frac:
        patch.max_size_frac = min(args.auto_max_size_frac, patch.max_size_frac * args.auto_size_multiplier)
        changed = changed or patch.max_size_frac > before_max_size

    if args.naturalness_weight > args.auto_min_naturalness_weight:
        args.naturalness_weight = max(
            args.auto_min_naturalness_weight,
            args.naturalness_weight * args.auto_naturalness_multiplier,
        )
        changed = changed or args.naturalness_weight < before_naturalness

    if args.auto_lr_multiplier > 1.0:
        args.lr = min(args.auto_max_lr, args.lr * args.auto_lr_multiplier)
        changed = changed or args.lr > before_lr

    event = {
        "target_index": target_index,
        "step": step,
        "reason": "auto_escalate",
        "actual_detection_score": actual_score,
        "actual_detection_class": actual_detection["class_name"] if actual_detection else None,
        "escalation_index": escalation_count + 1,
        "glare_count_before": before_count,
        "glare_count_after": patch.glare_count,
        "max_size_frac_before": before_max_size,
        "max_size_frac_after": patch.max_size_frac,
        "naturalness_weight_before": before_naturalness,
        "naturalness_weight_after": args.naturalness_weight,
        "lr_before": before_lr,
        "lr_after": args.lr,
    }
    return changed, event


def optimize_target(
    raw_model,
    detect_model,
    base_image_square: torch.Tensor,
    base_image_original: torch.Tensor,
    target: dict,
    target_box_square: torch.Tensor,
    target_box_original: torch.Tensor,
    args,
    target_index: int,
    device: str,
):
    patch = GradientLightPatch(
        glare_count=args.glare_count,
        min_size_frac=args.min_size_frac,
        max_size_frac=args.max_size_frac,
        seed=args.seed + target_index,
        device=device,
    )
    optimizer = torch.optim.Adam(patch.parameters(), lr=args.lr)
    best = {
        "score": float("inf"),
        "attack_score": float("inf"),
        "image": base_image_square.detach(),
        "glints": None,
        "state": None,
        "step": -1,
        "raw_score": None,
        "glare_count": patch.glare_count,
        "actual_detection_score": None,
    }
    progress = []
    growth_events = []
    last_improvement_step = 0
    last_growth_step = -args.growth_cooldown
    actual_check_count = 0
    best_actual_score = float("inf")
    last_actual_improvement_check = 0
    escalation_count = 0
    stop_reason = "step_budget"
    max_steps = args.max_steps if args.until_disappeared else args.steps
    unlimited_steps = args.until_disappeared and max_steps <= 0
    display_total = "unlimited" if unlimited_steps else str(max_steps)

    step = 0
    while unlimited_steps or step < max_steps:
        step_result = run_patch_step(
            raw_model,
            patch,
            optimizer,
            base_image_square,
            target,
            target_box_square,
            args,
        )
        attacked = step_result["attacked"]
        score_value = step_result["loss"]
        raw_value = step_result["raw_score"]
        progress.append(
            {
                "target_index": target_index,
                "target_class": target["class_name"],
                "step": step,
                "loss": score_value,
                "raw_score": raw_value,
                "naturalness": step_result["naturalness"],
                "relevant_predictions": step_result["relevant_predictions"],
                "glare_count": patch.glare_count,
                "actual_detection_score": None,
            }
        )
        improved = raw_value < best["attack_score"] - args.plateau_delta
        if raw_value < best["attack_score"]:
            best["score"] = score_value
            best["attack_score"] = raw_value
            best["image"] = attacked.detach()
            best["state"] = step_result["state"]
            best["glints"] = export_from_state(patch, step_result["state"], target_box_square)
            best["step"] = step
            best["raw_score"] = raw_value
            best["glare_count"] = patch.glare_count
        if improved:
            last_improvement_step = step

        plateaued = step - last_improvement_step >= args.plateau_window
        cooled_down = step - last_growth_step >= args.growth_cooldown
        can_grow = args.max_glare_count <= 0 or patch.glare_count < args.max_glare_count
        teleported = False
        if plateaued and cooled_down:
            if args.teleport_on_plateau and patch.glare_count > 0:
                teleport = try_teleport_weakest_glint(
                    raw_model,
                    patch,
                    base_image_square,
                    target,
                    target_box_square,
                    args,
                    target_index,
                    step,
                    best["attack_score"],
                    best["score"],
                )
                if teleport["improved"]:
                    optimizer = torch.optim.Adam(patch.parameters(), lr=args.lr)
                    last_growth_step = step
                    last_improvement_step = step
                    best["score"] = teleport["loss"]
                    best["attack_score"] = teleport["attack_score"]
                    best["image"] = teleport["image"]
                    best["glints"] = teleport["glints"]
                    best["state"] = teleport["state"]
                    best["step"] = step
                    best["raw_score"] = teleport["raw_score"]
                    best["glare_count"] = patch.glare_count
                    attacked = best["image"]
                    score_value = best["score"]
                    raw_value = best["raw_score"]
                    progress[-1]["loss"] = score_value
                    progress[-1]["raw_score"] = raw_value
                    progress[-1]["naturalness"] = teleport["naturalness"]
                    progress[-1]["relevant_predictions"] = teleport["relevant_predictions"]
                    progress[-1]["glare_count"] = patch.glare_count
                    growth_events.append(
                        {
                            "target_index": target_index,
                            "step": step,
                            "reason": "teleport",
                            "glare_count": patch.glare_count,
                            "teleported_glint": teleport["teleported_glint"],
                            "teleport_candidates": args.teleport_candidates,
                            "teleport_steps": args.teleport_steps,
                            "best_attack_score_before": teleport["best_attack_score_before"],
                            "best_attack_score_after": teleport["best_attack_score_after"],
                            "best_loss_before": teleport["best_loss_before"],
                            "best_loss_after": teleport["best_loss_after"],
                        }
                    )
                    teleported = True
                    print(
                        f"Plateau detected at step {step}; teleported glint "
                        f"{teleport['teleported_glint']} and improved "
                        f"raw score {teleport['best_attack_score_before']:.4f} "
                        f"-> {teleport['best_attack_score_after']:.4f}."
                    )

            if not teleported and can_grow:
                patch.add_glint(1)
                optimizer = torch.optim.Adam(patch.parameters(), lr=args.lr)
                last_growth_step = step
                last_improvement_step = step
                growth_event = {
                    "target_index": target_index,
                    "step": step,
                    "new_glare_count": patch.glare_count,
                    "best_loss": best["score"],
                    "best_attack_score": best["attack_score"],
                    "reason": "plateau",
                }
                growth_events.append(growth_event)
                max_text = "unlimited" if args.max_glare_count <= 0 else str(args.max_glare_count)
                print(
                    f"Plateau detected at step {step}; "
                    f"added glint {patch.glare_count}/{max_text}."
                )
            elif not teleported:
                last_growth_step = step
                if args.teleport_on_plateau:
                    growth_events.append(
                        {
                            "target_index": target_index,
                            "step": step,
                            "reason": "teleport_failed",
                            "glare_count": patch.glare_count,
                            "best_loss": best["score"],
                            "best_attack_score": best["attack_score"],
                            "teleport_candidates": args.teleport_candidates,
                            "teleport_steps": args.teleport_steps,
                        }
                    )
                    print(
                        f"Plateau detected at step {step}; teleport did not improve "
                        f"and glare count is capped at {patch.glare_count}."
                    )

        if args.until_disappeared and (step % args.check_every == 0 or step == max_steps - 1):
            with torch.no_grad():
                if teleported:
                    check_glints = best["glints"]
                else:
                    check_glints = export_from_state(patch, step_result["state"], target_box_square)
                check_original, _ = render_exported_glints(base_image_original, target_box_original, check_glints)
                check_bgr = tensor_rgb_to_bgr(check_original)
                check_result = run_yolo(
                    detect_model,
                    check_bgr,
                    device=device,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                )
                check_detections = detections_from_result(check_result)
                check_box = tuple(float(v) for v in target_box_original.detach().cpu().tolist())
                actual_score, actual_detection = max_any_detection_score(check_detections, check_box)
                progress[-1]["actual_detection_score"] = actual_score
                progress[-1]["actual_detection_class"] = (
                    actual_detection["class_name"] if actual_detection else None
                )
                actual_check_count += 1
                if actual_score < best_actual_score - args.actual_plateau_delta:
                    best_actual_score = actual_score
                    last_actual_improvement_check = actual_check_count
                print(
                    "Actual original-size YOLO check at step "
                    f"{step}: any-overlap score={actual_score:.4f} "
                    f"class={actual_detection['class_name'] if actual_detection else 'none'}"
                )
                if actual_score == 0.0:
                    best["score"] = score_value
                    best["attack_score"] = raw_value
                    best["image"] = attacked.detach()
                    if not teleported:
                        best["state"] = step_result["state"]
                    best["glints"] = check_glints
                    best["step"] = step
                    best["raw_score"] = raw_value
                    best["glare_count"] = patch.glare_count
                    best["actual_detection_score"] = actual_score
                    stop_reason = "disappeared"
                    break
                actual_stalled = (
                    args.auto_escalate
                    and actual_check_count - last_actual_improvement_check >= args.actual_plateau_checks
                )
                if actual_stalled:
                    escalated, auto_event = auto_escalate_patch(
                        patch,
                        args,
                        step,
                        target_index,
                        actual_score,
                        actual_detection,
                        escalation_count,
                    )
                    if escalated:
                        escalation_count += 1
                        optimizer = torch.optim.Adam(patch.parameters(), lr=args.lr)
                        last_growth_step = step
                        last_improvement_step = step
                        last_actual_improvement_check = actual_check_count
                        growth_events.append(auto_event)
                        progress[-1]["auto_escalated"] = True
                        progress[-1]["auto_escalation_index"] = escalation_count
                        progress[-1]["glare_count"] = patch.glare_count
                        print(
                            "Actual YOLO score stalled; auto-escalated "
                            f"glints {auto_event['glare_count_before']} -> {auto_event['glare_count_after']}, "
                            f"max_size {auto_event['max_size_frac_before']:.3f} -> "
                            f"{auto_event['max_size_frac_after']:.3f}, "
                            f"naturalness {auto_event['naturalness_weight_before']:.4f} -> "
                            f"{auto_event['naturalness_weight_after']:.4f}."
                        )
                    elif args.auto_stop_when_exhausted:
                        stop_reason = "auto_exhausted"
                        best["actual_detection_score"] = actual_score
                        break

        if step % args.print_every == 0 or (not unlimited_steps and step == max_steps - 1):
            print(
                f"[{step:04d}/{display_total}] "
                f"loss={score_value:.4f} raw={raw_value:.4f} "
                f"nat={step_result['naturalness']:.4f} glints={patch.glare_count}"
            )

        step += 1

    best["stop_reason"] = stop_reason
    best["steps_run"] = step + 1 if stop_reason == "disappeared" else step
    return best, progress, growth_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient-based natural light disappearance attack for YOLO.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--source-class", default=None)
    parser.add_argument("--region", type=parse_region, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--auto-attack",
        action="store_true",
        help="Use stronger self-tuning defaults and stop with a clear reason instead of requiring manual tuning.",
    )
    parser.add_argument("--until-disappeared", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Only used with --until-disappeared. 0 means no cap.")
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument(
        "--prune-glints",
        action="store_true",
        default=True,
        help="After a target disappears, remove any glints that are not needed to keep it disappeared.",
    )
    parser.add_argument(
        "--no-prune-glints",
        action="store_false",
        dest="prune_glints",
        help="Disable post-success glint pruning.",
    )
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--glare-count", type=int, default=5)
    parser.add_argument("--max-glare-count", type=int, default=12)
    parser.add_argument("--plateau-window", type=int, default=80)
    parser.add_argument("--plateau-delta", type=float, default=1e-3)
    parser.add_argument("--growth-cooldown", type=int, default=40)
    parser.add_argument(
        "--auto-escalate",
        action="store_true",
        default=True,
        help="When actual YOLO checks stall, automatically add capacity and relax naturalness constraints.",
    )
    parser.add_argument(
        "--no-auto-escalate",
        action="store_false",
        dest="auto_escalate",
        help="Disable actual-score-driven runtime escalation.",
    )
    parser.add_argument("--actual-plateau-checks", type=int, default=3)
    parser.add_argument("--actual-plateau-delta", type=float, default=0.02)
    parser.add_argument("--auto-glints-per-escalation", type=int, default=2)
    parser.add_argument("--auto-escalation-limit", type=int, default=8, help="0 means unlimited escalations.")
    parser.add_argument("--auto-max-size-frac", type=float, default=0.30)
    parser.add_argument("--auto-size-multiplier", type=float, default=1.25)
    parser.add_argument("--auto-min-naturalness-weight", type=float, default=0.0)
    parser.add_argument("--auto-naturalness-multiplier", type=float, default=0.55)
    parser.add_argument("--auto-lr-multiplier", type=float, default=1.15)
    parser.add_argument("--auto-max-lr", type=float, default=0.12)
    parser.add_argument(
        "--auto-stop-when-exhausted",
        action="store_true",
        default=True,
        help="Stop with stop_reason=auto_exhausted if escalation cannot continue.",
    )
    parser.add_argument(
        "--no-auto-stop-when-exhausted",
        action="store_false",
        dest="auto_stop_when_exhausted",
        help="Keep optimizing after auto escalation options are exhausted.",
    )
    parser.add_argument(
        "--teleport-on-plateau",
        action="store_true",
        default=True,
        help="When loss plateaus, try random relocations of the weakest glint before adding a new one.",
    )
    parser.add_argument(
        "--no-teleport-on-plateau",
        action="store_false",
        dest="teleport_on_plateau",
        help="Disable plateau relocation and only grow the glare pattern when possible.",
    )
    parser.add_argument(
        "--teleport-candidates",
        type=int,
        default=8,
        help="Random relocated positions to try for the weakest glint on each plateau.",
    )
    parser.add_argument(
        "--teleport-steps",
        type=int,
        default=20,
        help="Short Adam refinement steps for each relocated glint candidate.",
    )
    parser.add_argument(
        "--teleport-delta",
        type=float,
        default=5e-4,
        help="Minimum raw detector-score improvement required to accept a relocated glint.",
    )
    parser.add_argument("--naturalness-weight", type=float, default=0.08)
    parser.add_argument("--min-size-frac", type=float, default=0.025)
    parser.add_argument("--max-size-frac", type=float, default=0.16)
    parser.add_argument("--raw-iou", type=float, default=0.03)
    parser.add_argument(
        "--raw-topk",
        type=int,
        default=128,
        help="Optimize only the top-K relevant raw YOLO predictions. 0 uses all relevant predictions.",
    )
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    apply_auto_attack_defaults(args)
    args.check_every = max(1, args.check_every)
    args.print_every = max(1, args.print_every)
    args.plateau_window = max(1, args.plateau_window)
    args.growth_cooldown = max(1, args.growth_cooldown)
    args.actual_plateau_checks = max(1, args.actual_plateau_checks)
    args.actual_plateau_delta = max(0.0, args.actual_plateau_delta)
    args.auto_glints_per_escalation = max(1, args.auto_glints_per_escalation)
    args.auto_size_multiplier = max(1.0, args.auto_size_multiplier)
    args.auto_naturalness_multiplier = min(max(args.auto_naturalness_multiplier, 0.0), 1.0)
    args.auto_lr_multiplier = max(1.0, args.auto_lr_multiplier)
    args.teleport_candidates = max(1, args.teleport_candidates)
    args.teleport_steps = max(1, args.teleport_steps)
    args.teleport_delta = max(0.0, args.teleport_delta)
    if args.max_glare_count > 0 and args.max_glare_count < args.glare_count:
        args.max_glare_count = args.glare_count

    device = choose_device(args.device)
    describe_device(device)
    output_dir = ensure_dir(args.output or Path("outputs") / f"gradient_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    image_path = resolve_image_path(Path(args.image))
    image_bgr = load_image_bgr(image_path)
    height, width = image_bgr.shape[:2]

    print(f"Loading YOLO weights: {args.weights}")
    detect_yolo = YOLO(args.weights)
    detect_yolo.model.to(device).eval()

    # Keep the gradient model separate from Ultralytics' predict() path.
    # predict() uses inference-mode tensors internally, which cannot be reused in autograd.
    grad_yolo = YOLO(args.weights)
    grad_yolo.model.to(device).eval()
    for param in grad_yolo.model.parameters():
        param.requires_grad_(False)

    original_result = run_yolo(detect_yolo, image_bgr, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    original_detections = detections_from_result(original_result)
    if not original_detections:
        raise RuntimeError("YOLO found no detections in the original image. Try lowering --conf.")

    source_class_id = class_id_from_name(detect_yolo, args.source_class) if args.source_class else None
    targets = source_targets_from_detections(original_detections, source_class_id, args.region)
    if not targets:
        raise RuntimeError("No matching original detections to attack. Try lowering --conf or changing --source-class.")

    class_counts = {}
    for target in targets:
        class_counts[target["class_name"]] = class_counts.get(target["class_name"], 0) + 1
    print("Gradient attack targets:", ", ".join(f"{k} x{v}" for k, v in sorted(class_counts.items())))

    save_image(output_dir / "original.png", image_bgr)
    save_image(output_dir / "original_detections.jpg", original_result.plot())

    square_image = image_bgr_to_tensor_rgb(image_bgr, device=device, size=args.imgsz)
    current_square = square_image.detach()
    original_tensor = image_bgr_to_tensor_rgb(image_bgr, device=device, size=None)
    current_original = original_tensor.detach()
    all_progress = []
    all_growth_events = []
    all_prune_events = []
    patches = []

    for target_index, target in enumerate(targets):
        print(f"\nOptimizing target {target_index + 1}/{len(targets)}: {target['class_name']}")
        target_box_square = scale_box_to_square(target["xyxy"], width, height, args.imgsz, device)
        target_box_original = box_tensor(target["xyxy"], device)
        best, progress, growth_events = optimize_target(
            grad_yolo.model,
            detect_yolo,
            current_square,
            current_original,
            target,
            target_box_square,
            target_box_original,
            args,
            target_index,
            device,
        )
        prune_events = []
        if args.prune_glints and best["stop_reason"] == "disappeared":
            before_count = glint_count(best["glints"])
            best["glints"], prune_events, current_original = prune_successful_glints(
                detect_yolo,
                current_original,
                target_box_original,
                best["glints"],
                args,
                device,
            )
            if prune_events:
                after_count = glint_count(best["glints"])
                print(f"Pruned target glints {before_count} -> {after_count}.")
            best["glare_count"] = glint_count(best["glints"])
        else:
            current_original, _ = render_exported_glints(current_original, target_box_original, best["glints"])

        current_square, _ = render_exported_glints(current_square, target_box_square, best["glints"])
        current_square = current_square.detach()
        current_original = current_original.detach()
        all_progress.extend(progress)
        all_growth_events.extend(growth_events)
        for event in prune_events:
            all_prune_events.append(
                {
                    "target_index": target_index,
                    "target_class": target["class_name"],
                    **event,
                }
            )
        patches.append(
            {
                "target_index": target_index,
                "target": target,
                "best_step": best["step"],
                "best_loss": best["score"],
                "best_attack_score": best["attack_score"],
                "best_raw_score": best["raw_score"],
                "best_glare_count": best["glare_count"],
                "actual_detection_score": best["actual_detection_score"],
                "stop_reason": best["stop_reason"],
                "steps_run": best["steps_run"],
                "prune_events": prune_events,
                "pruned_glare_count": glint_count(best["glints"]),
                "glints": asdict(best["glints"]),
            }
        )

    attacked_original = original_tensor.detach()
    combined_mask = torch.zeros((height, width), device=device, dtype=torch.float32)
    for record in patches:
        target = record["target"]
        target_box = box_tensor(target["xyxy"], device)
        from torch_light_patch import ExportedGlints

        glints = ExportedGlints(**record["glints"])
        attacked_original, mask = render_exported_glints(attacked_original, target_box, glints)
        combined_mask = torch.maximum(combined_mask, mask)
        save_image(
            output_dir / f"patch_mask_target_{record['target_index']:02d}.png",
            cv2.cvtColor((mask.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        )

    attacked_bgr = tensor_rgb_to_bgr(attacked_original)
    save_image(output_dir / "attacked.png", attacked_bgr)
    save_image(output_dir / "attacked_square.png", tensor_rgb_to_bgr(current_square))
    save_image(
        output_dir / "patch_mask.png",
        cv2.cvtColor((combined_mask.detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
    )

    attacked_result = run_yolo(detect_yolo, attacked_bgr, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    attacked_detections = detections_from_result(attacked_result)
    save_image(output_dir / "attacked_detections.jpg", attacked_result.plot())

    for record in patches:
        target = record["target"]
        final_score, final_detection = max_any_detection_score(attacked_detections, tuple(target["xyxy"]))
        record["final_score"] = final_score
        record["final_detection"] = final_detection
        record["final_detection_class"] = final_detection["class_name"] if final_detection else None
        record["final_detection_class_id"] = final_detection["class_id"] if final_detection else None
        record["disappeared"] = final_score == 0.0

    write_progress(output_dir / "gradient_progress.csv", all_progress)
    write_progress(output_dir / "growth_events.csv", all_growth_events)
    write_progress(output_dir / "prune_events.csv", all_prune_events)
    summary = {
        "image": str(image_path),
        "weights": args.weights,
        "device": device,
        "attack": "gradient_disappear",
        "auto_attack": args.auto_attack,
        "imgsz": args.imgsz,
        "steps": args.steps,
        "until_disappeared": args.until_disappeared,
        "max_steps": args.max_steps,
        "check_every": args.check_every,
        "prune_glints": args.prune_glints,
        "lr": args.lr,
        "glare_count": args.glare_count,
        "max_glare_count": args.max_glare_count,
        "plateau_window": args.plateau_window,
        "plateau_delta": args.plateau_delta,
        "growth_cooldown": args.growth_cooldown,
        "auto_escalate": args.auto_escalate,
        "actual_plateau_checks": args.actual_plateau_checks,
        "actual_plateau_delta": args.actual_plateau_delta,
        "auto_glints_per_escalation": args.auto_glints_per_escalation,
        "auto_escalation_limit": args.auto_escalation_limit,
        "auto_max_size_frac": args.auto_max_size_frac,
        "auto_min_naturalness_weight": args.auto_min_naturalness_weight,
        "auto_stop_when_exhausted": args.auto_stop_when_exhausted,
        "teleport_on_plateau": args.teleport_on_plateau,
        "teleport_candidates": args.teleport_candidates,
        "teleport_steps": args.teleport_steps,
        "teleport_delta": args.teleport_delta,
        "growth_events": all_growth_events,
        "prune_events": all_prune_events,
        "naturalness_weight": args.naturalness_weight,
        "min_size_frac": args.min_size_frac,
        "max_size_frac": args.max_size_frac,
        "raw_topk": args.raw_topk,
        "source_class_id": source_class_id,
        "source_targets": targets,
        "image_fully_clear": len(attacked_detections) == 0,
        "success_count": sum(1 for record in patches if record["disappeared"]),
        "target_count": len(patches),
        "all_disappeared": all(record["disappeared"] for record in patches),
        "patches": patches,
        "original_detections": original_detections,
        "attacked_detections": attacked_detections,
    }
    with (output_dir / "attack_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nDone.")
    print(f"Disappeared targets: {summary['success_count']}/{summary['target_count']}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
