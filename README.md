# UPAR Challenge Baseline

PyTorch code for pedestrian attribute recognition over 40 attributes. The repository includes training, evaluation, and batch inference scripts, plus several baseline and hybrid model variants.

## Included Model Variants

The repository currently includes configs and/or factory branches for:

- `resnet50`
- `efficientnet_b0`
- `vit_b16`
- `swin_t`
- `pcb`
- `pcb_real`
- `hybrid_effb0_transformer`
- `hybrid_effb0_transformer_v3`
- `dual_branch_effb0_swin`
- `effb0_part`

The checked-in configs are in [`configs/`](/root/trunglm8/upar_challenge/configs).

## Metrics And Prediction Rules

Validation and evaluation report:

- loss
- exact-match accuracy
- label accuracy
- macro F1
- mAP
- per-class AP
- per-class accuracy

Post-processing is group-aware rather than pure independent thresholding:

- `argmax` groups: age, hair, upper-body color, lower-body color, lower-body type
- thresholded at `0.5`: gender, upper-body length, lower-body length, backpack, bag, glasses, hat

## Attribute List

`Age-Young`, `Age-Adult`, `Age-Old`, `Gender-Female`, `Hair-Length-Short`, `Hair-Length-Long`, `Hair-Length-Bald`, `UpperBody-Length-Short`, `UpperBody-Color-Black`, `UpperBody-Color-Blue`, `UpperBody-Color-Brown`, `UpperBody-Color-Green`, `UpperBody-Color-Grey`, `UpperBody-Color-Orange`, `UpperBody-Color-Pink`, `UpperBody-Color-Purple`, `UpperBody-Color-Red`, `UpperBody-Color-White`, `UpperBody-Color-Yellow`, `UpperBody-Color-Other`, `LowerBody-Length-Short`, `LowerBody-Color-Black`, `LowerBody-Color-Blue`, `LowerBody-Color-Brown`, `LowerBody-Color-Green`, `LowerBody-Color-Grey`, `LowerBody-Color-Orange`, `LowerBody-Color-Pink`, `LowerBody-Color-Purple`, `LowerBody-Color-Red`, `LowerBody-Color-White`, `LowerBody-Color-Yellow`, `LowerBody-Color-Other`, `LowerBody-Type-Trousers&Shorts`, `LowerBody-Type-Skirt&Dress`, `Accessory-Backpack`, `Accessory-Bag`, `Accessory-Glasses-Normal`, `Accessory-Glasses-Sun`, `Accessory-Hat`

## Repository Layout

```text
.
├── configs/        # YAML experiment configs
├── outputs/        # Training runs and checkpoints
├── raw_data/       # Local datasets and CSV annotations in this workspace
├── scripts/        # Small helper shell scripts
├── src/
│   ├── data/       # Dataset loader and image transforms
│   ├── engine/     # Losses and trainer
│   ├── models/     # Model definitions and factory
│   └── utils/      # Metrics, checkpoints, seeds, class weights
├── evaluate.py
├── inference.py
├── train.py
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

[`src/data/dataset.py`](/root/trunglm8/upar_challenge/src/data/dataset.py) loads a CSV with:

- column 1: image path
- remaining columns: numeric labels or metadata

Current loader behavior matters:

- It resolves images by `basename`, not by the full relative path.
- It scans `data.img_dirs` in order and picks the first matching filename.
- Rows whose image filename is not found are silently skipped.
- It reads labels with `iloc[:, 1:-1]`, so the last CSV column is ignored by the current code.
- Most checked-in configs use `skiprows: 1`, which assumes a header row exists.

Example row from [`raw_data/test.csv`](/root/trunglm8/upar_challenge/raw_data/test.csv):

```csv
MEVID/images/0.jpg,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
```

## Configuration

The training entrypoint expects a config shaped like this:

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
  skiprows: 1

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
  lr: 0.0005
  weight_decay: 0.001
  step_size: 10
  gamma: 0.1
  scheduler: step
```

Model-specific options currently used by the factory:

- `pcb`, `pcb_real`: `num_parts`, `reduced_dim`
- `hybrid_effb0_transformer`, `hybrid_effb0_transformer_v3`: `backbone_name`, `d_model`, `nhead`, `num_transformer_layers`, `dim_feedforward`, `transformer_dropout`
- `hybrid_effb0_transformer_v3`: `num_parts`
- `dual_branch_effb0_swin`: `cnn_name`, `transformer_name`, `fusion_dim`

## Training

Update the dataset paths in a config first, then run:

```bash
python train.py --config configs/resnet50.yaml
```

Helper scripts:

```bash
bash scripts/train_resnet.sh
bash scripts/train_vit.sh
```

Each run is written under `outputs/<experiment_name>/` and currently includes:

- `best_model.pth`
- `last_model.pth`
- `history.json`
- `curve_loss.png`
- `curve_map.png`
- `curve_f1_macro.png`
- `curve_exact_match_acc.png`
- `curve_lr.png`
- `best_model_per_class_accuracy.png`

## Evaluation

[`evaluate.py`](/root/trunglm8/upar_challenge/evaluate.py) expects `data.test_csv` to be a labeled split:

```bash
python evaluate.py \
  --config configs/resnet50.yaml \
  --checkpoint outputs/resnet50_baseline_withcw/best_model.pth
```

Helper script:

```bash
bash scripts/eval.sh
```

## Inference

[`inference.py`](/root/trunglm8/upar_challenge/inference.py) currently runs on a directory of images, not a single file:

```bash
python inference.py \
  --config configs/resnet50.yaml \
  --checkpoint outputs/resnet50_baseline_withcw/best_model.pth \
  --image_dir /path/to/images \
  --num_images 20 \
  --random_sample
```

The script:

- loads up to `num_images` from `image_dir`
- prints grouped positive predictions per image
- saves annotated visualizations to `test_result/<model_name>/vis_20/`

## Notes On This Checkout

This workspace already contains local data under [`raw_data/`](/root/trunglm8/upar_challenge/raw_data), plus several previous runs under [`outputs/`](/root/trunglm8/upar_challenge/outputs).

Important caveats in the current repository state:

- Most configs use absolute paths tied to this workspace.
- Several configs reference `raw_data/train.csv` and `raw_data/val.csv`, which are not present in the repo.
- Configs still reference `raw_data/data/PETA/images`, but that directory is not present here.
- [`configs/vit_b16.yaml`](/root/trunglm8/upar_challenge/configs/vit_b16.yaml) currently sets `model.name: vit_s16`, while the factory supports `vit_b16`.
- [`configs/effb0_part.yaml`](/root/trunglm8/upar_challenge/configs/effb0_part.yaml) uses a different schema from what [`train.py`](/root/trunglm8/upar_challenge/train.py) expects.
- [`src/models/factory.py`](/root/trunglm8/upar_challenge/src/models/factory.py) has an `effb0_part` branch but does not currently import `EfficientNetB0PartAttrModel`.
- Training applies fixed label smoothing (`epsilon = 0.1`) inside the trainer.

## Quick Start

1. Install dependencies.
2. Fix the dataset paths in one config.
3. Verify `skiprows` and your CSV column layout.
4. Train with `python train.py --config ...`.
5. Evaluate with `python evaluate.py --config ... --checkpoint ...`.
6. Run directory inference with `python inference.py --config ... --checkpoint ... --image_dir ...`.
