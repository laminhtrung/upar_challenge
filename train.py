import os
import yaml
import argparse
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.class_weights import compute_class_weights

from src.data.dataset import MultiFolderDataset
from src.data.transforms import build_transforms
from src.models.factory import build_model
from src.engine.losses import build_loss
from src.engine.trainer import Trainer
from src.utils.seed import set_seed


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


def plot_history(history, save_path):
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 3, 2)
    plt.plot(history["mAP"], label="mAP")
    plt.plot(history["f1_macro"], label="f1_macro")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.legend()
    plt.title("mAP / F1")

    plt.subplot(1, 3, 3)
    plt.plot(history["label_acc"], label="label_acc")
    plt.plot(history["exact_match_acc"], label="exact_match_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(config_path):
    cfg = load_yaml(config_path)
    set_seed(cfg.get("seed", 42))

    if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda":
        gpu_id = cfg.get("gpu_id", 0)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"[Device] Using GPU: cuda:{gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU")
    

    run_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    os.makedirs(run_dir, exist_ok=True)

    train_tf = build_transforms(tuple(cfg["data"]["image_size"]), is_train=True)
    val_tf = build_transforms(tuple(cfg["data"]["image_size"]), is_train=False)

    train_dataset = MultiFolderDataset(
        csv_file=cfg["data"]["train_csv"],
        img_dirs=cfg["data"]["img_dirs"],
        transform=train_tf,
        skiprows=cfg["data"].get("skiprows", 1),
    )

    val_dataset = MultiFolderDataset(
        csv_file=cfg["data"]["val_csv"],
        img_dirs=cfg["data"]["img_dirs"],
        transform=val_tf,
        skiprows=cfg["data"].get("skiprows", 1),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )

    pos_weight = cfg.get("loss", {}).get("pos_weight", None)
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
        num_parts=cfg["model"].get("num_parts", 6),
        reduced_dim=cfg["model"].get("reduced_dim", 256),
    ).to(device)

    class_weights = None
    if cfg.get("loss", {}).get("use_auto_class_weight", False):
        class_weights = compute_class_weights(
            train_loader=train_loader,
            num_classes=cfg["num_classes"],
            device=device
        )
        print("[Loss] Auto class_weights =", class_weights)
        
    elif cfg.get("loss", {}).get("class_weight", None) is not None:
        class_weights = torch.tensor(
            cfg["loss"]["class_weight"],
            dtype=torch.float32,
            device=device
        )
        print("[Loss] Config class_weights =", class_weights)
    
    criterion = build_loss(
        loss_name=cfg["loss"]["name"],
        pos_weight=None,
        weight=class_weights
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["train"]["step_size"],
        gamma=cfg["train"]["gamma"]
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_names=CLASS_NAMES,
        output_dir=run_dir
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["train"]["epochs"]
    )

    plot_history(history, os.path.join(run_dir, "training_curves.png"))
    print(f"[Done] Results saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)