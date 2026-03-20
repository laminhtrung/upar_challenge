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

from src.utils.pos_weight import compute_pos_weight


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


import os
import matplotlib.pyplot as plt


def plot_loss_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_map_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["mAP"], label="mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("mAP Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_f1_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["f1_macro"], label="F1 Macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Macro Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_exact_match_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["exact_match_acc"], label="Exact Match Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Exact Match Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_best_class_accuracy(history, class_names, save_path):
    best_acc_per_class = history.get("best_acc_per_class", None)
    best_epoch = history.get("best_epoch", None)

    if best_acc_per_class is None:
        print("[Warning] No best_acc_per_class found in history.")
        return

    plt.figure(figsize=(16, 8))
    plt.bar(range(len(class_names)), best_acc_per_class)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"Per-class Accuracy at Best Model (Epoch {best_epoch})")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_lr_curve(history, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history["lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
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

    pos_weight = None
    if cfg.get("loss", {}).get("use_pos_weight", False):
        pos_weight = compute_pos_weight(
            train_loader=train_loader,
            num_classes=cfg["num_classes"],
            device=device,
            max_value=cfg.get("loss", {}).get("pos_weight_max", 10.0)
        )
        print("[Loss] Auto pos_weight =", pos_weight)
        print(
            "[Loss] pos_weight stats -> "
            f"min: {pos_weight.min().item():.4f}, "
            f"max: {pos_weight.max().item():.4f}, "
            f"mean: {pos_weight.mean().item():.4f}"
        )

    model = build_model(
        model_name=cfg["model"]["name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.5),
        num_parts=cfg["model"].get("num_parts", 6),
        reduced_dim=cfg["model"].get("reduced_dim", 256),
        backbone_name=cfg["model"].get("backbone_name", "efficientnet_b0"),
        d_model=cfg["model"].get("d_model", 256),
        nhead=cfg["model"].get("nhead", 8),
        num_transformer_layers=cfg["model"].get("num_transformer_layers", 2),
        dim_feedforward=cfg["model"].get("dim_feedforward", 512),
        transformer_dropout=cfg["model"].get("transformer_dropout", 0.1),
        cnn_name=cfg["model"].get("cnn_name", "efficientnet_b0"),
        transformer_name=cfg["model"].get("transformer_name", "swin_tiny_patch4_window7_224"),
        fusion_dim=cfg["model"].get("fusion_dim", 256),
        image_size=tuple(cfg["data"]["image_size"]),
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
        pos_weight=pos_weight,
        weight=class_weights,
        device=device
    )
    print(criterion)
    print("class_weights:", class_weights)
    print(cfg["loss"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scheduler_name = cfg["train"].get("scheduler", "step")
    print("scheduler_name =", scheduler_name)
    
    
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["train"]["epochs"]),
            eta_min=float(cfg["train"].get("eta_min", 1e-6))
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg["train"]["step_size"]),
            gamma=cfg["train"]["gamma"]
        )
    print("scheduler =", scheduler)
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

    plot_loss_curve(history, os.path.join(run_dir, "curve_loss.png"))
    plot_map_curve(history, os.path.join(run_dir, "curve_map.png"))
    plot_f1_curve(history, os.path.join(run_dir, "curve_f1_macro.png"))
    plot_exact_match_curve(history, os.path.join(run_dir, "curve_exact_match_acc.png"))
    plot_lr_curve(history, os.path.join(run_dir, "curve_lr.png"))
    plot_best_class_accuracy(
        history,
        CLASS_NAMES,
        os.path.join(run_dir, "best_model_per_class_accuracy.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)