# YOLO Light Patch Attack Prototype

This is a standalone prototype for testing whether light-shaped image patches can make YOLO-style object detections disappear.

The current attack is digital-only: it draws a soft elliptical light patch on each target object, runs YOLO, and searches for patch parameters that suppress that object's original detection.

## Setup

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

For this laptop's Blackwell GPU, PyTorch should be a CUDA build that supports `sm_120`, such as the nightly `cu128` build:

```powershell
python -m pip install --pre --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Quick Test

Run a disappearance attack on every detected object in the image. The script optimizes one light patch per object and clips each patch to that object's bounding box:

```powershell
python attack_yolo.py --image cartestimage.avif --weights yolov8n.pt --device cuda:0
```

The first run downloads `yolov8n.pt` automatically.

Outputs are written to `outputs/run_<timestamp>/`:

- `original.png`
- `original_detections.jpg`
- `attacked.png`
- `attacked_detections.jpg`
- `patch_mask.png`
- `progress.csv`
- `attack_summary.json`

## Useful Commands

Suppress every detection of a specific class, with one clipped patch per matching object:

```powershell
python attack_yolo.py --image cartestimage.avif --source-class car --device cuda:0
```

Use a stronger search when confidence barely moves:

```powershell
python attack_yolo.py --image cartestimage.avif --source-class car --iterations 1000 --candidates-per-iter 16 --score-conf 0.01 --device cuda:0
```

Use several smaller, less obvious glares per object:

```powershell
python attack_yolo.py --image cartestimage.avif --source-class car --iterations 1200 --candidates-per-iter 16 --glare-count 5 --max-size-frac 0.14 --naturalness-weight 0.08 --device cuda:0
```

Attack only a known region:

```powershell
python attack_yolo.py --image cartestimage.avif --source-class car --region 100,120,420,360 --device cuda:0
```

## Notes

This is a research prototype. The attack is successful for a target when YOLO no longer detects that original class overlapping the original box above the configured confidence threshold.

`--conf` controls what counts as a final visible detection. `--score-conf` is lower by default so the optimizer can still see weak detections and push them down further.

Smaller glare patches are more natural-looking, but they usually need more iterations than one large glare.

The script defaults to `--iterations 800` and `--candidates-per-iter 12`. Use `--device cuda:0` on an NVIDIA CUDA machine, or `--device mps` on an Apple Silicon machine with PyTorch MPS available. With `--device auto`, it will try CUDA, then MPS, then CPU.

For faster experiments, reduce YOLO input size or attack fewer targets:

```powershell
python attack_yolo.py --image cartestimage.avif --source-class person --iterations 400 --candidates-per-iter 8 --imgsz 416 --device cuda:0
```

Candidates are evaluated in batches, so increasing `--candidates-per-iter` is usually more efficient than running many separate commands.

## Gradient Attack

`gradient_light_attack.py` is the gradient-based version. It keeps YOLO frozen and optimizes only natural-light glint parameters with Adam:

```powershell
python gradient_light_attack.py --image cartestimage.avif --weights yolov8n.pt --source-class car --steps 500 --glare-count 5 --imgsz 640 --device cuda:0
```

For a stronger run:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --steps 1200 --lr 0.04 --glare-count 6 --max-glare-count 12 --max-size-frac 0.18 --naturalness-weight 0.06 --imgsz 640 --device cuda:0
```

The gradient attack can grow its light pattern when loss plateaus. It starts with `--glare-count`, then adds one glint at a time until `--max-glare-count` or the step budget is reached:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --steps 1800 --glare-count 5 --max-glare-count 14 --plateau-window 80 --plateau-delta 0.001 --growth-cooldown 40 --device cuda:0
```

On each plateau, the optimizer first tries to escape a local minimum by relocating the weakest glint and briefly refining several candidates. A relocation is accepted only when it improves the raw detector score for the target object; if none improve that attack score, it grows the glare pattern when allowed:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --steps 1800 --glare-count 5 --max-glare-count 14 --teleport-candidates 12 --teleport-steps 30 --teleport-delta 0.0005 --device cuda:0
```

For large targets, `--raw-topk` keeps the differentiable loss focused on the highest-scoring raw YOLO predictions instead of spreading gradient over thousands of weak predictions:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --steps 1800 --raw-topk 128 --device cuda:0
```

To keep optimizing until the object disappears, use `--until-disappeared`. A practical capped run:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --until-disappeared --max-steps 5000 --check-every 50 --glare-count 5 --max-glare-count 18 --plateau-window 80 --plateau-delta 0.001 --growth-cooldown 40 --device cuda:0
```

Disappearance means no YOLO detection of any class overlaps the original target box. If the original `person` becomes a different class such as `tie`, the target is not counted as disappeared. The summary also reports `image_fully_clear` for whether the whole final image has zero detections.

After a target disappears, the gradient attack prunes unnecessary glints by removing one at a time and keeping each removal only if the target still has no overlapping detection. Use `--no-prune-glints` to keep the first successful glare pattern unchanged.

For an uncapped run, set both caps to `0` and stop manually with `Ctrl+C` if needed:

```powershell
python gradient_light_attack.py --image inputs\your_image.webp --weights yolov8n.pt --source-class person --until-disappeared --max-steps 0 --max-glare-count 0 --check-every 50 --device cuda:0
```

Outputs are saved under `outputs/gradient_run_<timestamp>/`. The key files are `attacked.png`, `attacked_detections.jpg`, `patch_mask.png`, `gradient_progress.csv`, `growth_events.csv`, and `attack_summary.json`.
