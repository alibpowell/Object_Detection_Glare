# Successful Experiment Commands

These commands are retained as reference points from successful or useful
experiments. They are starting points, not guaranteed results: model versions,
hardware, images, and random seeds can all affect the outcome.

## Person example

```bash
python gradient_light_attack.py \
  --image inputs/jlo.webp \
  --weights yolov8n.pt \
  --source-class person \
  --until-disappeared \
  --max-steps 5000 \
  --glare-count 6 \
  --max-glare-count 0 \
  --max-size-frac 0.05 \
  --naturalness-weight 0.30 \
  --lr 0.03 \
  --device cuda:0
```

## Car baseline

```bash
python gradient_light_attack.py \
  --image inputs/car1.jpg \
  --weights yolov8n.pt \
  --source-class car \
  --auto-attack \
  --polish-iterations 800 \
  --polish-candidates 12 \
  --polish-max-size-frac 0.40 \
  --polish-max-opacity 0.90 \
  --polish-max-intensity 1.8 \
  --polish-max-glare-count 80 \
  --device cuda:0
```
