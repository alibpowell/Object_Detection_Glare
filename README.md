# YOLO Light-Glare Disappearance Attack

This repository is a research prototype for testing whether synthetic, natural-looking light glare can suppress YOLO detections in still images.

The main workflow is `gradient_light_attack.py`. It freezes a YOLO model, optimizes differentiable glare/glint parameters with PyTorch, checks the actual postprocessed YOLO detections on the original-size image, optionally polishes the result against real YOLO outputs, and prunes unnecessary glints after success.

There is also an older black-box search path in `attack_yolo.py`. It is useful for simpler experiments, but the gradient attack is the current focus.

## Repository Layout

`gradient_light_attack.py`
: Current primary attack. Optimizes warm elliptical glints with gradients, escapes plateaus, checks real YOLO detections, auto-escalates when actual scores stall, polishes against postprocessed YOLO, prunes extra glints, and writes full artifacts.

`torch_light_patch.py`
: Differentiable PyTorch glare renderer. It owns the learnable glint parameters: center, radius, angle, opacity, intensity, and warm RGB tint.

`yolo_raw_loss.py`
: Differentiable loss over raw YOLO predictions. The current loss suppresses the highest-scoring prediction of any class in the target region, not just the original class.

`attack_yolo.py`
: Older non-gradient optimizer. It samples and mutates glare parameters, evaluates candidate images through YOLO, and keeps the best candidate.

`optimize_patch.py`
: Core black-box candidate-search implementation used by `attack_yolo.py`.

`light_patch.py`
: NumPy/OpenCV renderer for the older black-box light patch path.

`sweep_params.py`
: Batch runner for randomized `gradient_light_attack.py` parameter sweeps. It records ranked trial summaries under `outputs/sweeps/...`.

`yolo_utils.py`
: YOLO loading, prediction, class lookup, IoU, and detection conversion helpers.

`image_io.py`
: Image loading and saving utilities, including Pillow fallback for AVIF/WebP-like formats.

`inputs/`
: Example images.

`outputs/`
: Generated run artifacts. This directory is ignored by git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The first YOLO run will download `yolov8n.pt` if it is not already present.

### GPU Devices

Device selection is controlled with `--device`.

Use automatic selection:

```bash
--device auto
```

`auto` tries CUDA first, then Apple MPS, then CPU.

Force NVIDIA CUDA:

```bash
--device cuda:0
```

Force Apple Silicon MPS:

```bash
--device mps
```

CPU works, but gradient attacks can be slow. If a run appears to crawl, check the startup line. It prints either `CUDA device: ...`, `Apple GPU device: MPS`, or `Running on CPU.`

## Recommended Command

For most images, start with auto attack mode:

```bash
python gradient_light_attack.py \
  --image inputs/car1.jpg \
  --weights yolov8n.pt \
  --source-class car \
  --auto-attack \
  --device auto
```

For the example person image:

```bash
python gradient_light_attack.py \
  --image inputs/jlo.webp \
  --weights yolov8n.pt \
  --source-class person \
  --auto-attack \
  --device auto
```

`--auto-attack` is the least manual mode. It enables disappearance checking, uses stronger capped defaults, escalates when actual YOLO checks stall, runs actual-YOLO polishing, and prunes unnecessary glints after success.

It does not guarantee every image/model combination will disappear. If the search exhausts its escalation budget, the run stops honestly with `stop_reason: "auto_exhausted"` instead of running forever.

## Success Definition

For the gradient attack, a target is counted as disappeared only when:

```text
no YOLO detection of any class overlaps the original target box
```

This matters. If the original class was `person` and the attacked image produces an overlapping `tie`, that is not success. The summary will show:

```json
"disappeared": false,
"final_detection_class": "tie"
```

The summary also reports:

```json
"image_fully_clear": true
```

only when the final attacked image has zero YOLO detections anywhere.

## Gradient Attack Pipeline

`gradient_light_attack.py` runs one target at a time.

1. Load the input image.
2. Run YOLO on the original image.
3. Select targets by `--source-class` and optional `--region`.
4. Create a differentiable square working tensor at `--imgsz`.
5. Optimize glint parameters with Adam while YOLO weights stay frozen.
6. Use raw YOLO predictions for gradient loss.
7. Periodically run actual YOLO prediction on the original-size rendered image.
8. If the actual score stalls, auto-escalate when enabled.
9. If gradient optimization plateaus, teleport weak glints or add glints.
10. Optionally polish exported glints directly against postprocessed YOLO.
11. If the target disappears, prune glints that are not needed.
12. Save final original-size outputs and JSON/CSV diagnostics.

### Differentiable Loss

The raw loss suppresses the top raw YOLO predictions overlapping the target box. It is class-agnostic inside the target region, so it suppresses replacement classes too.

Important knobs:

```bash
--raw-topk 128
--temperature 0.03
--raw-iou 0.03
```

`--raw-topk` prevents huge targets from spreading gradient over thousands of weak predictions. Lower values focus harder on the strongest predictions; `0` uses all relevant predictions.

### Actual YOLO Checks

With `--until-disappeared`, the script runs full YOLO prediction every `--check-every` steps on the original-size image that will actually be saved.

The check uses any overlapping class:

```text
any-overlap score = max confidence of any YOLO detection overlapping the original target box
```

The target stops only when that score is `0.0`.

### Plateau Escape

The gradient optimizer can get stuck. The script has two local-minimum escape paths:

`teleport`
: Finds the weakest glint, randomizes it, briefly refines several candidates, and accepts a candidate only if the raw attack score improves.

`growth`
: Adds another glint when teleport does not help and `--max-glare-count` allows growth.

Relevant knobs:

```bash
--plateau-window 80
--plateau-delta 0.001
--growth-cooldown 40
--teleport-candidates 8
--teleport-steps 20
--teleport-delta 0.0005
--max-glare-count 12
```

Use `--no-teleport-on-plateau` to disable teleporting.

### Auto Escalation

Auto escalation watches the real postprocessed YOLO score, not just the differentiable loss. If actual YOLO checks stop improving for `--actual-plateau-checks`, it can:

- add glints,
- increase maximum allowed glint size,
- reduce naturalness pressure,
- increase learning rate within a cap.

Relevant knobs:

```bash
--auto-escalate
--actual-plateau-checks 3
--actual-plateau-delta 0.02
--auto-glints-per-escalation 2
--auto-escalation-limit 8
--auto-max-size-frac 0.30
--auto-size-multiplier 1.25
--auto-min-naturalness-weight 0.0
--auto-naturalness-multiplier 0.55
--auto-lr-multiplier 1.15
--auto-max-lr 0.12
```

Use `--no-auto-escalate` to disable this. Use `--no-auto-stop-when-exhausted` if you want the optimizer to keep going after escalation is exhausted.

### Actual-YOLO Polish

After gradient optimization, `--polish-actual` is enabled by default. This stage mutates exported glints directly and evaluates each candidate through actual postprocessed YOLO.

This is slower than gradient steps, but it helps when the differentiable raw loss and final NMS behavior disagree.

Relevant knobs:

```bash
--polish-iterations 300
--polish-candidates 8
--polish-delta 0.0001
--polish-max-size-frac 0.30
--polish-max-opacity 0.75
--polish-max-intensity 1.4
--polish-max-glare-count 60
--polish-add-probability 0.20
```

Use `--no-polish-actual` to disable this stage. Use `--no-polish-add-glints` to prevent polish from adding glints.

### Glint Pruning

After a target disappears, `--prune-glints` is enabled by default. It removes one glint at a time and keeps a removal only if the target still has no overlapping detection.

This does not prove a mathematically minimal patch, but it usually reduces unnecessary glints from the first successful pattern.

Use:

```bash
--no-prune-glints
```

to keep the first successful pattern unchanged.

## Output Files

Gradient attack outputs are written to:

```text
outputs/gradient_run_<timestamp>/
```

Main files:

`original.png`
: Original input image.

`original_detections.jpg`
: YOLO detections on the original image.

`attacked.png`
: Final original-size attacked image. This is the main image to inspect.

`attacked_detections.jpg`
: YOLO detections on `attacked.png`.

`attacked_square.png`
: Square optimizer working image at `--imgsz`. This is a debug artifact, not the final result.

`patch_mask.png`
: Combined glare mask over all attacked targets.

`patch_mask_target_XX.png`
: Per-target glare mask.

`gradient_progress.csv`
: Per-step differentiable loss, raw score, naturalness, glint count, and actual YOLO check fields when available.

`growth_events.csv`
: Plateau growth, teleport, and auto-escalation events.

`polish_events.csv`
: Accepted actual-YOLO polish mutations.

`prune_events.csv`
: Glints removed after success.

`attack_summary.json`
: Full run configuration, original detections, final detections, target records, exported glints, success counts, and stop reasons.

## Reading `attack_summary.json`

Important top-level fields:

`success_count`
: Number of targets with no overlapping final detection of any class.

`target_count`
: Number of attacked targets.

`all_disappeared`
: Whether every attacked target disappeared.

`image_fully_clear`
: Whether the final image has zero YOLO detections anywhere.

`attacked_detections`
: Full YOLO detections on the final image.

Important per-target fields under `patches`:

`stop_reason`
: Why optimization stopped for that target.

Common values:

```text
disappeared
step_budget
auto_exhausted
```

`final_score`
: Highest final YOLO confidence of any class overlapping the original target box.

`final_detection_class`
: Class name of that overlapping final detection, if any.

`disappeared`
: `true` only when `final_score` is `0.0`.

`best_glare_count`
: Glint count before or after later stages, depending on the final selected state.

`pruned_glare_count`
: Final glint count after pruning.

`glints`
: Exported normalized glint parameters used to render the final patch.

## Common Commands

Attack one class with auto mode:

```bash
python gradient_light_attack.py --image inputs/car1.jpg --weights yolov8n.pt --source-class car --auto-attack --device auto
```

Attack all original detections:

```bash
python gradient_light_attack.py --image inputs/car1.jpg --weights yolov8n.pt --auto-attack --device auto
```

Attack only a known region:

```bash
python gradient_light_attack.py --image inputs/car1.jpg --weights yolov8n.pt --source-class car --region 100,120,420,360 --auto-attack
```

Run a capped high-effort attack:

```bash
python gradient_light_attack.py \
  --image inputs/car1.jpg \
  --weights yolov8n.pt \
  --source-class car \
  --until-disappeared \
  --max-steps 5000 \
  --check-every 25 \
  --glare-count 5 \
  --max-glare-count 40 \
  --max-size-frac 0.20 \
  --naturalness-weight 0.03 \
  --polish-iterations 800 \
  --polish-candidates 12 \
  --device auto
```

Keep the glare more conservative:

```bash
python gradient_light_attack.py \
  --image inputs/car1.jpg \
  --weights yolov8n.pt \
  --source-class car \
  --auto-attack \
  --max-size-frac 0.08 \
  --auto-max-size-frac 0.16 \
  --naturalness-weight 0.20 \
  --polish-max-size-frac 0.20 \
  --polish-max-opacity 0.50
```

Let it run without a step cap:

```bash
python gradient_light_attack.py \
  --image inputs/jlo.webp \
  --weights yolov8n.pt \
  --source-class person \
  --until-disappeared \
  --max-steps 0 \
  --max-glare-count 0 \
  --check-every 50 \
  --device auto
```

Use this carefully. `--max-steps 0` with `--until-disappeared` means unlimited steps.

## Parameter Sweeps

Use `sweep_params.py` when one image/class needs multiple randomized trials.

Example:

```bash
python sweep_params.py \
  --image inputs/car1.jpg \
  --source-class car \
  --weights yolov8n.pt \
  --device auto \
  --profile balanced \
  --stop-after-successes 1
```

Profiles:

`smoke`
: Few quick trials.

`balanced`
: Default general sweep.

`focused`
: More constrained glare sizes with higher naturalness pressure.

`minimal`
: Searches for smaller/subtler glare patterns.

`deep`
: Larger, slower sweep.

Sweep outputs are written under:

```text
outputs/sweeps/car_sweep_<timestamp>/
```

Important files:

`results.csv`
: Ranked trial table.

`best_trial.json`
: Best row from `results.csv`.

Each trial directory also contains that trial's normal `gradient_light_attack.py` artifacts and `run.log`.

## Older Black-Box Attack

`attack_yolo.py` is the earlier search-based attack. It does not use gradients. Instead, it generates candidate glare patches, runs YOLO on each candidate, and keeps the best.

Example:

```bash
python attack_yolo.py --image inputs/car1.jpg --weights yolov8n.pt --source-class car --device auto
```

Stronger search:

```bash
python attack_yolo.py \
  --image inputs/car1.jpg \
  --weights yolov8n.pt \
  --source-class car \
  --iterations 1200 \
  --candidates-per-iter 16 \
  --glare-count 5 \
  --max-size-frac 0.14 \
  --naturalness-weight 0.08 \
  --device auto
```

Important difference: the older black-box path scores disappearance by the original class, not by any overlapping class. The gradient path is stricter and should be preferred for current experiments.

## Troubleshooting

`ModuleNotFoundError: No module named 'cv2'`
: Activate `.venv` and install requirements.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`Running on CPU.`
: PyTorch did not see CUDA or MPS. Use `--device cuda:0` or `--device mps` only if your installed PyTorch supports that backend.

`WARNING NMS time limit exceeded`
: YOLO spent too long filtering many candidate boxes. Try raising `--conf`, lowering `--imgsz`, or checking less often with a larger `--check-every`.

The run never stops
: With `--until-disappeared`, `--max-steps 0` means unlimited. Use `--auto-attack` or set an explicit cap with `--max-steps`.

The object changes class instead of disappearing
: The current gradient attack treats this as failure. Check `final_detection_class` in `attack_summary.json`.

`attacked_square.png` is square
: Expected. It is the optimizer's square debug image. Inspect `attacked.png` and `attacked_detections.jpg` for the final original-size result.

## Notes and Limitations

This is a digital-only research prototype. It edits pixels directly; it does not model a physical projector, camera response, exposure, lens flare, print/display constraints, or viewpoint changes.

No command can guarantee disappearance for every image and every YOLO model. The current auto mode is designed to generalize better by escalating and polishing automatically, but it will still fail on some inputs.

The gradient attack optimizes on a square resized tensor, then exports normalized glints back to the original image. Actual success is always checked on the original-size rendered image.

Generated artifacts in `outputs/`, downloaded model weights such as `*.pt`, virtual environments, and caches are ignored by git.
