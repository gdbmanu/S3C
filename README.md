# S3C — Self-Supervised Space Construction with foveated vision

Self-supervised learning with foveated vision and saccadic eye movements,
built on DINOv1 ViT-Base/8.

## Architecture

- **FoveatedMultiViT** : ViT with foveal pyramid mosaic input (4 scales, 2×2)
- **Multiple views on the same scene** 
- **Teacher-student EMA** : cross-saccade feature prediction with shift regression

## Installation

```bash
git clone https://github.com/yourname/S3C
cd S3C
pip install -e .
```

## Training

```bash
# Linear probe on ImageNet
python scripts/train_linear.py --config configs/linear_probe.yaml

# Teacher-student with saccades
python scripts/train_teacher_student.py \
    --config configs/teacher_student.yaml \
    training.epochs=30 training.lr=1e-4
```

## Notebooks

- `01_saccades_visualization` — visualize saccade sequences on images
- `02_pos_embed_sanity_check` — verify foveal pos embed coherence
- `03_results_analysis`       — plot training curves and evaluate features

## Checkpoints

Checkpoints are saved locally in `./checkpoints*/` and never tracked by git.
