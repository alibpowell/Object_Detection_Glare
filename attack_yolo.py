from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from image_io import ensure_dir, load_image_bgr, save_image
from light_patch import render_light_mask
from optimize_patch import AttackConfig, optimize_light_patch
from yolo_utils import (
    box_iou,
    class_id_from_name,
    detections_from_result,
    load_yolo,
    max_detection_score,
    run_yolo,
)


def parse_region(value: str | None):
    if value is None:
        return None
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region must be x1,y1,x2,y2")
    return tuple(parts)


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> None:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        index = int(device.split(":")[1]) if ":" in device else 0
        print(f"CUDA device: {torch.cuda.get_device_name(index)}")
        print(f"Torch CUDA: {torch.version.cuda}")
    else:
        print("Running on CPU.")


def write_progress(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_targets_from_detections(
    detections: list[dict],
    source_class_id: int | None,
    region: tuple[float, float, float, float] | None,
) -> list[dict]:
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


def mean_source_score(detections: list[dict], targets: list[dict]) -> float:
    if not targets:
        return 0.0
    scores = [
        max_detection_score(detections, target["class_id"], tuple(target["xyxy"]))
        for target in targets
    ]
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize light patches that make YOLO detections disappear.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Ultralytics YOLO weights, e.g. yolov8n.pt.")
    parser.add_argument(
        "--source-class",
        default=None,
        help="Class to make disappear. If omitted, attacks every original detection with one patch per object.",
    )
    parser.add_argument("--region", type=parse_region, default=None, help="Optional x1,y1,x2,y2 region to attack.")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--candidates-per-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for reported detections.")
    parser.add_argument(
        "--score-conf",
        type=float,
        default=0.01,
        help="Lower confidence threshold used inside the optimizer for smoother attack scoring.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--glare-count", type=int, default=4, help="Small glare spots to optimize per object.")
    parser.add_argument("--naturalness-weight", type=float, default=0.05)
    parser.add_argument("--min-size-frac", type=float, default=0.025)
    parser.add_argument("--max-size-frac", type=float, default=0.18)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument("--output", default=None, help="Output directory. Defaults to outputs/run_<timestamp>.")
    args = parser.parse_args()

    device = choose_device(args.device)
    describe_device(device)
    image_path = Path(args.image)
    output_dir = ensure_dir(args.output or Path("outputs") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print(f"Loading image: {image_path}")
    image = load_image_bgr(image_path)
    print(f"Loading YOLO weights: {args.weights} on {device}")
    model = load_yolo(args.weights, device=device)

    original_result = run_yolo(model, image, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    original_detections = detections_from_result(original_result)
    if not original_detections:
        raise RuntimeError("YOLO found no detections in the original image. Try lowering --conf.")

    region = args.region
    source_class_id = class_id_from_name(model, args.source_class) if args.source_class else None

    source_targets = source_targets_from_detections(original_detections, source_class_id, region)
    if not source_targets:
        raise RuntimeError("No matching original detections to attack. Try lowering --conf or changing --source-class.")

    class_counts = {}
    for target in source_targets:
        class_counts[target["class_name"]] = class_counts.get(target["class_name"], 0) + 1
    target_text = ", ".join(f"{name} x{count}" for name, count in sorted(class_counts.items()))
    print(f"Disappearance attack targets: {len(source_targets)} detections ({target_text})")
    print("Each target gets its own patch clipped to that object's box.")

    baseline_source_score = mean_source_score(original_detections, source_targets)

    save_image(output_dir / "original.png", image)
    save_image(output_dir / "original_detections.jpg", original_result.plot())

    progress = []
    patch_records = []
    combined_mask = np.zeros(image.shape[:2], dtype=np.float32)

    attacked_image = image.copy()
    for target_index, target in enumerate(source_targets):
        target_region = tuple(target["xyxy"])
        print(
            "\nOptimizing target "
            f"{target_index + 1}/{len(source_targets)}: "
            f"{target['class_name']} at {tuple(round(v, 1) for v in target_region)}"
        )
        config = AttackConfig(
            mode="disappear",
            source_class_id=target["class_id"],
            target_class_id=None,
            region=target_region,
            source_targets=[target],
            iterations=args.iterations,
            seed=args.seed + target_index,
            conf=args.score_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            clip_patch_to_region=True,
            candidates_per_iter=args.candidates_per_iter,
            glare_count=args.glare_count,
            naturalness_weight=args.naturalness_weight,
            min_size_frac=args.min_size_frac,
            max_size_frac=args.max_size_frac,
        )
        result = optimize_light_patch(model, attacked_image, config)
        attacked_image = result["best_image"]

        target_mask = render_light_mask(image.shape, result["best_params"], clip_region=target_region)
        combined_mask = np.maximum(combined_mask, target_mask)
        save_image(
            output_dir / f"patch_mask_target_{target_index:02d}.png",
            cv2.cvtColor((target_mask * 255).astype("uint8"), cv2.COLOR_GRAY2BGR),
        )

        for row in result["progress"]:
            progress.append(
                {
                    "target_index": target_index,
                    "target_class": target["class_name"],
                    **row,
                }
            )
        patch_records.append(
            {
                "target_index": target_index,
                "target": target,
                "best_optimizer_score": result["best_score"],
                "patch": result["best_params"].to_dict(),
            }
        )

    attacked_result = run_yolo(model, attacked_image, device=device, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    attacked_detections = detections_from_result(attacked_result)
    final_source_score = mean_source_score(attacked_detections, source_targets)
    for record in patch_records:
        target = record["target"]
        final_score = max_detection_score(attacked_detections, target["class_id"], tuple(target["xyxy"]))
        record["final_score"] = final_score
        record["disappeared"] = final_score == 0.0
    mask_vis = (combined_mask * 255).astype("uint8")

    save_image(output_dir / "attacked.png", attacked_image)
    save_image(output_dir / "attacked_detections.jpg", attacked_result.plot())
    save_image(output_dir / "patch_mask.png", cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR))
    write_progress(output_dir / "progress.csv", progress)

    summary = {
        "image": str(image_path),
        "weights": args.weights,
        "device": device,
        "attack": "disappear",
        "source_class_id": source_class_id,
        "region": list(region) if region else None,
        "report_conf": args.conf,
        "score_conf": args.score_conf,
        "candidates_per_iter": args.candidates_per_iter,
        "glare_count": args.glare_count,
        "naturalness_weight": args.naturalness_weight,
        "min_size_frac": args.min_size_frac,
        "max_size_frac": args.max_size_frac,
        "source_targets": source_targets,
        "baseline_source_score": baseline_source_score,
        "final_source_score": final_source_score,
        "success_count": sum(1 for record in patch_records if record["disappeared"]),
        "target_count": len(patch_records),
        "all_disappeared": all(record["disappeared"] for record in patch_records),
        "patches": patch_records,
        "original_detections": original_detections,
        "attacked_detections": attacked_detections,
    }
    with (output_dir / "attack_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nDone.")
    print(f"Original source score: {baseline_source_score:.4f}")
    print(f"Final source score:    {final_source_score:.4f}")
    print(f"Disappeared targets:   {sum(1 for record in patch_records if record['disappeared'])}/{len(patch_records)}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
