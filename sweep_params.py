from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


FIELDNAMES = [
    "rank",
    "trial",
    "status",
    "disappeared",
    "image_fully_clear",
    "final_score",
    "final_detection_class",
    "original_confidence",
    "best_glare_count",
    "pruned_glare_count",
    "glint_area_proxy",
    "mean_opacity",
    "mean_intensity",
    "steps_run",
    "stop_reason",
    "runtime_sec",
    "timeout_min",
    "output_dir",
    "seed",
    "lr",
    "glare_count",
    "max_glare_count",
    "max_size_frac",
    "naturalness_weight",
    "auto_max_size_frac",
    "auto_escalation_limit",
    "polish_iterations",
    "polish_candidates",
    "polish_max_size_frac",
    "polish_max_opacity",
    "polish_max_intensity",
    "polish_max_glare_count",
    "polish_add_probability",
    "returncode",
    "error",
]


PROFILE_DEFAULTS = {
    "smoke": {
        "trials": 4,
        "max_steps": 700,
        "polish_iterations": (150, 350),
        "polish_candidates": [6, 8],
    },
    "balanced": {
        "trials": 16,
        "max_steps": 1800,
        "polish_iterations": (300, 800),
        "polish_candidates": [8, 10, 12],
    },
    "focused": {
        "trials": 10,
        "max_steps": 1200,
        "polish_iterations": (350, 750),
        "polish_candidates": [8, 10, 12],
    },
    "minimal": {
        "trials": 12,
        "max_steps": 1800,
        "polish_iterations": (450, 900),
        "polish_candidates": [8, 10, 12],
    },
    "deep": {
        "trials": 32,
        "max_steps": 3500,
        "polish_iterations": (600, 1500),
        "polish_candidates": [10, 12, 16],
    },
}


def sample_log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_trial(rng: random.Random, trial: int, profile: str, args) -> dict:
    profile_defaults = PROFILE_DEFAULTS[profile]
    polish_low, polish_high = profile_defaults["polish_iterations"]
    if profile == "focused":
        max_size_frac = rng.choice([0.045, 0.055, 0.065, 0.08, 0.10])
        auto_max_size_frac = max(max_size_frac, rng.choice([0.16, 0.18, 0.20, 0.22]))
        polish_max_size_frac = max(auto_max_size_frac, rng.choice([0.25, 0.30, 0.34, 0.38]))
        max_glare_count = rng.choice([48, 56, 64, 70])
        polish_max_glare_count = max(max_glare_count, rng.choice([60, 64, 70, 80]))

        return {
            "seed": args.seed + trial,
            "lr": round(sample_log_uniform(rng, 0.03, 0.065), 5),
            "glare_count": rng.choice([5, 6, 7]),
            "max_glare_count": max_glare_count,
            "max_size_frac": max_size_frac,
            "naturalness_weight": rng.choice([0.15, 0.20, 0.25, 0.30, 0.35]),
            "auto_max_size_frac": auto_max_size_frac,
            "auto_escalation_limit": rng.choice([5, 6, 8]),
            "polish_iterations": rng.randint(polish_low, polish_high),
            "polish_candidates": rng.choice(profile_defaults["polish_candidates"]),
            "polish_max_size_frac": polish_max_size_frac,
            "polish_max_opacity": rng.choice([0.55, 0.60, 0.65, 0.70, 0.75]),
            "polish_max_intensity": rng.choice([1.3, 1.45, 1.6, 1.7]),
            "polish_max_glare_count": polish_max_glare_count,
            "polish_add_probability": rng.choice([0.08, 0.12, 0.16, 0.20]),
        }

    if profile == "minimal":
        max_size_frac = rng.choice([0.025, 0.032, 0.04, 0.05, 0.06])
        auto_max_size_frac = max(max_size_frac, rng.choice([0.08, 0.10, 0.12, 0.15]))
        polish_max_size_frac = max(auto_max_size_frac, rng.choice([0.10, 0.14, 0.18, 0.22, 0.26]))
        max_glare_count = rng.choice([12, 16, 20, 24, 30, 36])
        polish_max_glare_count = max(max_glare_count, rng.choice([16, 20, 24, 30, 36, 44]))

        return {
            "seed": args.seed + trial,
            "lr": round(sample_log_uniform(rng, 0.025, 0.08), 5),
            "glare_count": rng.choice([2, 3, 4, 5]),
            "max_glare_count": max_glare_count,
            "max_size_frac": max_size_frac,
            "naturalness_weight": rng.choice([0.25, 0.35, 0.45, 0.60, 0.80]),
            "auto_max_size_frac": auto_max_size_frac,
            "auto_escalation_limit": rng.choice([2, 3, 4, 5]),
            "polish_iterations": rng.randint(polish_low, polish_high),
            "polish_candidates": rng.choice(profile_defaults["polish_candidates"]),
            "polish_max_size_frac": polish_max_size_frac,
            "polish_max_opacity": rng.choice([0.30, 0.40, 0.50, 0.60]),
            "polish_max_intensity": rng.choice([0.7, 0.9, 1.1, 1.3]),
            "polish_max_glare_count": polish_max_glare_count,
            "polish_add_probability": rng.choice([0.02, 0.05, 0.08, 0.12]),
        }

    max_size_frac = rng.choice([0.045, 0.06, 0.08, 0.11, 0.14, 0.18])
    auto_max_size_frac = max(max_size_frac, rng.choice([0.16, 0.22, 0.30, 0.38, 0.45]))
    polish_max_size_frac = max(auto_max_size_frac, rng.choice([0.22, 0.30, 0.38, 0.46]))
    max_glare_count = rng.choice([32, 48, 64, 80, 100, 120])
    polish_max_glare_count = max(max_glare_count, rng.choice([60, 80, 100, 140]))

    return {
        "seed": args.seed + trial,
        "lr": round(sample_log_uniform(rng, 0.025, 0.12), 5),
        "glare_count": rng.choice([4, 5, 6, 8]),
        "max_glare_count": max_glare_count,
        "max_size_frac": max_size_frac,
        "naturalness_weight": rng.choice([0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35]),
        "auto_max_size_frac": auto_max_size_frac,
        "auto_escalation_limit": rng.choice([4, 6, 8, 10]),
        "polish_iterations": rng.randint(polish_low, polish_high),
        "polish_candidates": rng.choice(profile_defaults["polish_candidates"]),
        "polish_max_size_frac": polish_max_size_frac,
        "polish_max_opacity": rng.choice([0.55, 0.65, 0.75, 0.85, 0.95]),
        "polish_max_intensity": rng.choice([1.2, 1.45, 1.7, 2.0, 2.4]),
        "polish_max_glare_count": polish_max_glare_count,
        "polish_add_probability": rng.choice([0.12, 0.16, 0.20, 0.25]),
    }


def read_summary(output_dir: Path) -> dict:
    with (output_dir / "attack_summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def glint_stats(patches: list[dict]) -> dict:
    radii = []
    opacities = []
    intensities = []
    for patch in patches:
        glints = patch.get("glints") or {}
        radii.extend(glints.get("radius") or [])
        opacities.extend(glints.get("opacity") or [])
        intensities.extend(glints.get("intensity") or [])

    area_proxy = 0.0
    for radius_pair, opacity in zip(radii, opacities):
        if len(radius_pair) == 2:
            area_proxy += math.pi * float(radius_pair[0]) * float(radius_pair[1]) * float(opacity)

    return {
        "glint_area_proxy": area_proxy,
        "mean_opacity": sum(opacities) / len(opacities) if opacities else 0.0,
        "mean_intensity": sum(intensities) / len(intensities) if intensities else 0.0,
    }


def summarize_trial(
    trial: int,
    params: dict,
    output_dir: Path,
    runtime_sec: float,
    returncode: int,
    error: str,
    status: str | None = None,
    timeout_min: float = 0.0,
) -> dict:
    row = {
        "rank": "",
        "trial": trial,
        "status": status or ("failed" if returncode else "ok"),
        "runtime_sec": round(runtime_sec, 2),
        "timeout_min": timeout_min,
        "output_dir": str(output_dir),
        "returncode": returncode,
        "error": error,
        **params,
    }

    if returncode:
        return row

    try:
        summary = read_summary(output_dir)
        patches = summary.get("patches") or []
        first_patch = patches[0] if patches else {}
        target = first_patch.get("target") or {}
        stats = glint_stats(patches)
        row.update(
            {
                "disappeared": bool(summary.get("all_disappeared")),
                "image_fully_clear": bool(summary.get("image_fully_clear")),
                "final_score": first_patch.get("final_score"),
                "final_detection_class": first_patch.get("final_detection_class"),
                "original_confidence": target.get("original_confidence"),
                "best_glare_count": first_patch.get("best_glare_count"),
                "pruned_glare_count": first_patch.get("pruned_glare_count"),
                "steps_run": first_patch.get("steps_run"),
                "stop_reason": first_patch.get("stop_reason"),
                **stats,
            }
        )
    except Exception as exc:  # Keep the sweep alive even if one run writes a bad summary.
        row["status"] = "summary_error"
        row["error"] = str(exc)

    return row


def ranking_key(row: dict) -> tuple:
    disappeared = 1 if row.get("disappeared") is True else 0
    image_clear = 1 if row.get("image_fully_clear") is True else 0
    final_score = row.get("final_score")
    final_score = float(final_score) if final_score not in ("", None) else 999.0
    glare_count = row.get("pruned_glare_count") or row.get("best_glare_count") or 999
    area_proxy = row.get("glint_area_proxy") or 999.0
    runtime_sec = row.get("runtime_sec") or 999999.0
    return (-disappeared, -image_clear, final_score, int(glare_count), float(area_proxy), float(runtime_sec))


def write_results(path: Path, rows: list[dict]) -> None:
    ranked = sorted(rows, key=ranking_key)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)


def build_command(args, params: dict, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "gradient_light_attack.py",
        "--image",
        args.image,
        "--weights",
        args.weights,
        "--source-class",
        args.source_class,
        "--auto-attack",
        "--until-disappeared",
        "--max-steps",
        str(args.max_steps),
        "--check-every",
        str(args.check_every),
        "--device",
        args.device,
        "--output",
        str(output_dir),
        "--seed",
        str(params["seed"]),
        "--lr",
        str(params["lr"]),
        "--glare-count",
        str(params["glare_count"]),
        "--max-glare-count",
        str(params["max_glare_count"]),
        "--max-size-frac",
        str(params["max_size_frac"]),
        "--naturalness-weight",
        str(params["naturalness_weight"]),
        "--auto-max-size-frac",
        str(params["auto_max_size_frac"]),
        "--auto-escalation-limit",
        str(params["auto_escalation_limit"]),
        "--polish-iterations",
        str(params["polish_iterations"]),
        "--polish-candidates",
        str(params["polish_candidates"]),
        "--polish-max-size-frac",
        str(params["polish_max_size_frac"]),
        "--polish-max-opacity",
        str(params["polish_max_opacity"]),
        "--polish-max-intensity",
        str(params["polish_max_intensity"]),
        "--polish-max-glare-count",
        str(params["polish_max_glare_count"]),
        "--polish-add-probability",
        str(params["polish_add_probability"]),
        "--print-every",
        str(args.print_every),
    ]
    if args.no_prune:
        command.append("--no-prune-glints")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run randomized parameter sweeps for gradient_light_attack.py.")
    parser.add_argument("--image", default=r"inputs\car1.jpg")
    parser.add_argument("--source-class", default="car")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=sorted(PROFILE_DEFAULTS), default="balanced")
    parser.add_argument("--trials", type=int, default=0, help="Overrides the selected profile's trial count.")
    parser.add_argument("--max-steps", type=int, default=0, help="Overrides the selected profile's step cap.")
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--stop-after-successes", type=int, default=0)
    parser.add_argument(
        "--trial-timeout-min",
        type=float,
        default=20.0,
        help="Kill and record a trial as timeout after this many minutes. Use 0 for no timeout.",
    )
    args = parser.parse_args()

    profile_defaults = PROFILE_DEFAULTS[args.profile]
    args.trials = args.trials or profile_defaults["trials"]
    args.max_steps = args.max_steps or profile_defaults["max_steps"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or Path("outputs") / "sweeps" / f"car_sweep_{stamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.csv"
    rows = []
    rng = random.Random(args.seed)

    print(f"Writing sweep results to: {output_root}")
    print(f"Trials: {args.trials} | profile={args.profile} | max_steps={args.max_steps}")

    for trial in range(1, args.trials + 1):
        params = sample_trial(rng, trial, args.profile, args)
        trial_dir = output_root / f"trial_{trial:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, params, trial_dir)
        log_path = trial_dir / "run.log"

        print(
            f"\nTrial {trial}/{args.trials}: "
            f"lr={params['lr']} glints={params['glare_count']} "
            f"max_size={params['max_size_frac']} nat={params['naturalness_weight']} "
            f"polish={params['polish_iterations']}x{params['polish_candidates']}"
        )

        started = time.perf_counter()
        error = ""
        status = None
        returncode = 0
        with log_path.open("w", encoding="utf-8") as log:
            log.write("Command:\n")
            log.write(" ".join(command) + "\n\n")
            log.flush()
            try:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.trial_timeout_min * 60 if args.trial_timeout_min > 0 else None,
                )
                returncode = completed.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
                returncode = -9
                error = f"Trial exceeded --trial-timeout-min {args.trial_timeout_min}"
                log.write(f"\n\n{error}\n")
        runtime_sec = time.perf_counter() - started
        if returncode and not error:
            error = f"See {log_path}"

        row = summarize_trial(
            trial,
            params,
            trial_dir,
            runtime_sec,
            returncode,
            error,
            status=status,
            timeout_min=args.trial_timeout_min,
        )
        rows.append(row)
        write_results(results_path, rows)

        final_score = row.get("final_score")
        print(
            f"Trial {trial} status={row['status']} disappeared={row.get('disappeared')} "
            f"final_score={final_score} runtime={runtime_sec / 60:.1f} min"
        )

        if args.stop_after_successes:
            successes = sum(1 for row in rows if row.get("disappeared") is True)
            if successes >= args.stop_after_successes:
                print(f"Stopping after {successes} successful trial(s).")
                break

    write_results(results_path, rows)
    best = sorted(rows, key=ranking_key)[0] if rows else None
    if best:
        with (output_root / "best_trial.json").open("w", encoding="utf-8") as handle:
            json.dump(best, handle, indent=2)
        print("\nBest trial:")
        print(
            f"trial={best['trial']} disappeared={best.get('disappeared')} "
            f"final_score={best.get('final_score')} output={best.get('output_dir')}"
        )
    print(f"\nMaster results: {results_path}")


if __name__ == "__main__":
    main()
