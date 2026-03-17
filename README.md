# UPAR Challenge Baseline

PyTorch training code for pedestrian attribute recognition with 40 binary / grouped-multiclass attributes. The repository trains a single model over mixed person-image sources and includes scripts for training, evaluation, and single-image inference.

Supported backbones:

- `resnet50`
- `efficientnet_b0`
- `vit_b16`
- `swin_t`
- `pcb` (a ResNet50-based part pooling variant)

## What The Code Does

The project builds a 40-logit attribute predictor and evaluates it with:

- validation loss
- exact-match accuracy
- per-label accuracy
- macro F1
- mean average precision (mAP)
- per-class AP / accuracy

Prediction post-processing is not plain multilabel thresholding for every class. Some attribute groups are decoded with `argmax` as mutually exclusive categories:

- age: 3 classes
- hair: 3 classes
- upper-body color: 12 classes
- lower-body color: 12 classes
- lower-body type: 2 classes
- glasses: 2 classes

The remaining attributes are thresholded at `0.5`:

- gender
- upper-body sleeve length
- lower-body length
- backpack
- bag
- hat

## Attribute List

The model predicts these 40 labels:

`Age-Young`, `Age-Adult`, `Age-Old`, `Gender-Female`, `Hair-Length-Short`, `Hair-Length-Long`, `Hair-Length-Bald`, `UpperBody-Length-Short`, `UpperBody-Color-Black`, `UpperBody-Color-Blue`, `UpperBody-Color-Brown`, `UpperBody-Color-Green`, `UpperBody-Color-Grey`, `UpperBody-Color-Orange`, `UpperBody-Color-Pink`, `UpperBody-Color-Purple`, `UpperBody-Color-Red`, `UpperBody-Color-White`, `UpperBody-Color-Yellow`, `UpperBody-Color-Other`, `LowerBody-Length-Short`, `LowerBody-Color-Black`, `LowerBody-Color-Blue`, `LowerBody-Color-Brown`, `LowerBody-Color-Green`, `LowerBody-Color-Grey`, `LowerBody-Color-Orange`, `LowerBody-Color-Pink`, `LowerBody-Color-Purple`, `LowerBody-Color-Red`, `LowerBody-Color-White`, `LowerBody-Color-Yellow`, `LowerBody-Color-Other`, `LowerBody-Type-Trousers&Shorts`, `LowerBody-Type-Skirt&Dress`, `Accessory-Backpack`, `Accessory-Bag`, `Accessory-Glasses-Normal`, `Accessory-Glasses-Sun`, `Accessory-Hat`

## Repository Layout

```text
.
├── configs/                 # YAML configs for each backbone
├── scripts/                 # Small helper shell scripts
├── src/
│   ├── data/                # Dataset loader and image transforms
│   ├── engine/              # Losses and training loop
│   ├── models/              # Model definitions and factory
│   └── utils/               # Metrics, checkpointing, seeding, class weights
├── train.py                 # Training entrypoint
├── evaluate.py              # Evaluation entrypoint
├── inference.py             # Single-image inference entrypoint
├── requirements.txt
└── raw_data/                # Local data/annotations currently present in this workspace
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

`src/data/dataset.py` expects a CSV where:

- column 1 is the image path
- columns 2..41 are numeric labels

Example:

```csv
Market1501/bounding_box_train/0002_c1s1_000451_03.jpg,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0
```

Important loader behavior:

- The loader strips the directory part with `os.path.basename(...)` and then searches each directory in `data.img_dirs` for a matching filename.
- Because of that, the first column can be a relative dataset path, but matching ultimately depends on the filename only.
- If two datasets contain the same basename, the first matching directory in `img_dirs` wins.
- `skiprows` defaults to `1` in the configs. If your CSV has no header row, set `skiprows: 0` or the first sample will be skipped.

## Data In This Workspace

This checkout already contains a large `raw_data/` tree, including:

- `raw_data/data/Market1501/...`
- `raw_data/data/PA100k/...`
- `raw_data/data/phase1/train/train_0.csv`, `train_1.csv`, `train_2.csv`
- `raw_data/data/phase1/val_task1/val_all.csv`
- `raw_data/data/phase2/annotations/val_gt/val_gt.csv`
- `raw_data/data/phase2/annotations/test_task2/test_imgs.csv`
- `raw_data/data/phase2/annotations/test_task2/test_queries.csv`
- `raw_data/test.csv`

Current config caveats:

- `configs/efficientnet_b0.yaml` points to `/root/trunglm8/upar_challenge/raw_data/train.csv` and `/root/trunglm8/upar_challenge/raw_data/val.csv`, but those files are not present in this repo.
- The other configs point to old Kaggle-style absolute paths and must be edited before running locally.
- Configs reference a `PETA/images` directory, but that directory is not present in this workspace.
- Some YAML files contain `use_auto_pos_weight`, but the training code currently uses `loss.pos_weight`, `loss.class_weight`, or `loss.use_auto_class_weight`.

## Configuration

Each config in `configs/*.yaml` follows this structure:

```yaml
experiment_name: resnet50_baseline
seed: 42
num_classes: 40
device: cuda
gpu_id: 0
output_dir: outputs

data:
  train_csv: /path/to/train.csv
  val_csv: /path/to/val.csv
  test_csv: /path/to/test.csv
  img_dirs:
    - /path/to/images_a
    - /path/to/images_b
  image_size: [384, 128]
  batch_size: 32
  num_workers: 4
  skiprows: 0

model:
  name: resnet50
  pretrained: true
  dropout: 0.5

loss:
  name: bce
  use_auto_class_weight: false
  class_weight: null
  pos_weight: null

train:
  epochs: 25
  lr: 0.001
  weight_decay: 0.001
  step_size: 10
  gamma: 0.1
```

Model-specific options:

- `pcb` also supports `num_parts` and `reduced_dim`
- `vit_b16` and `swin_t` are configured for `224x224`
- `resnet50`, `efficientnet_b0`, and `pcb` are configured for `384x128`

## Training

Edit one config first so all paths match your machine, then run:

```bash
python train.py --config configs/resnet50.yaml
```

Helper scripts:

```bash
bash scripts/train_resnet.sh
bash scripts/train_vit.sh
```

Training outputs are written to:

```text
outputs/<experiment_name>/
├── best_model.pth
├── last_model.pth
├── history.json
└── training_curves.png
```

## Evaluation

`evaluate.py` expects `data.test_csv` in the config to point to a labeled split.

```bash
python evaluate.py \
  --config configs/resnet50.yaml \
  --checkpoint outputs/resnet50_baseline/best_model.pth
```

The included helper script is:

```bash
bash scripts/eval.sh
```

## Inference

Run single-image inference with a trained checkpoint:

```bash
python inference.py \
  --config configs/resnet50.yaml \
  --checkpoint outputs/resnet50_baseline/best_model.pth \
  --image /path/to/image.jpg
```

The script prints one line per attribute with binary prediction and probability.

## Model Notes

- `resnet50`: torchvision ResNet-50 backbone with a BN-ReLU-Dropout-Linear head
- `efficientnet_b0`: torchvision EfficientNet-B0 with a replaced classifier
- `vit_b16`: timm `vit_base_patch16_224`
- `swin_t`: timm `swin_tiny_patch4_window7_224`
- `pcb`: ResNet-50 feature extractor with adaptive vertical part pooling and a shared attribute head

## Implementation Notes

- Loss: `BCEWithLogitsLoss`
- Optimizer: `AdamW`
- Scheduler: `StepLR`
- Augmentation: resize, random rotation, random horizontal flip, normalize
- Checkpointing: best model is selected by validation `mAP`
- Reproducibility: seed is set for Python, NumPy, and PyTorch; cuDNN runs in deterministic mode

## Known Limitations

- Several config files are templates and are not runnable without path fixes.
- The repo does not include a script to merge or generate final `train.csv` / `val.csv` files.
- Empty placeholder modules exist in `src/configs.py`, `src/engine/evaluator.py`, `src/models/cnn_heads.py`, `src/utils/common.py`, and `src/utils/logger.py`.
- The dataset loader silently drops rows whose image basename cannot be found in any configured image directory.

## Quick Start Checklist

1. Install dependencies from `requirements.txt`.
2. Update one YAML config so `train_csv`, `val_csv`, `test_csv`, and `img_dirs` point to real local files.
3. Set `skiprows` correctly for your CSV format.
4. Train with `python train.py --config ...`.
5. Evaluate with `python evaluate.py --config ... --checkpoint ...`.
6. Run `inference.py` for single-image inspection.
