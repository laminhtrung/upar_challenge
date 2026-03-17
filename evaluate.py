import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultiFolderDataset
from src.data.transforms import build_transforms
from src.models.factory import build_model
from src.engine.losses import build_loss
from src.engine.trainer import Trainer
from src.utils.checkpoint import load_checkpoint


CLASS_NAMES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short",
    "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short", "UpperBody-Color-Black",
    "UpperBody-Color-Blue", "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple", "UpperBody-Color-Red",
    "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other", "LowerBody-Length-Short",
    "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink", "LowerBody-Color-Purple",
    "LowerBody-Color-Red", "LowerBody-Color-White", "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path, checkpoint_path):
    cfg = load_yaml(config_path)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    test_tf = build_transforms(tuple(cfg["data"]["image_size"]), is_train=False)

    test_dataset = MultiFolderDataset(
        csv_file=cfg["data"]["test_csv"],
        img_dirs=cfg["data"]["img_dirs"],
        transform=test_tf,
        skiprows=cfg["data"].get("skiprows", 1),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        dropout=cfg["model"].get("dropout", 0.5),
        num_parts=cfg["model"].get("num_parts", 6),
        reduced_dim=cfg["model"].get("reduced_dim", 256),
    ).to(device)

    model = load_checkpoint(model, checkpoint_path, device=device)

    criterion = build_loss(cfg["loss"]["name"])

    evaluator = Trainer(
        model=model,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
        device=device,
        class_names=CLASS_NAMES,
        output_dir="."
    )

    metrics = evaluator.validate(test_loader)

    print("\n[Test Results]")
    print(f"Val Loss        : {metrics['val_loss']:.4f}")
    print(f"Exact Match Acc : {metrics['exact_match_acc']:.4f}")
    print(f"Label Accuracy  : {metrics['label_acc']:.4f}")
    print(f"F1 Macro        : {metrics['f1_macro']:.4f}")
    print(f"mAP             : {metrics['mAP']:.4f}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}: Acc={metrics['acc_per_class'][i]:.4f}, AP={metrics['ap_per_class'][i]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)