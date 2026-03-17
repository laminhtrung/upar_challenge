import os
import json
import torch
from tqdm import tqdm
from src.utils.metrics import (
    postprocess_predictions,
    calculate_map,
    calculate_f1,
    exact_match_accuracy,
    label_accuracy,
    per_class_accuracy,
)
from src.utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, class_names, output_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = class_names
        self.output_dir = output_dir

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "exact_match_acc": [],
            "label_acc": [],
            "f1_macro": [],
            "mAP": []
        }

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs, preds = postprocess_predictions(logits)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

        all_probs = torch.cat(all_probs, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        val_loss = total_loss / len(loader.dataset)
        exact_acc = exact_match_accuracy(all_preds, all_labels)
        lab_acc = label_accuracy(all_preds, all_labels)
        f1 = calculate_f1(all_preds, all_labels)
        map_value, ap_per_class = calculate_map(all_probs, all_labels, all_labels.shape[1])
        acc_per_class = per_class_accuracy(all_preds, all_labels)

        return {
            "val_loss": val_loss,
            "exact_match_acc": exact_acc,
            "label_acc": lab_acc,
            "f1_macro": f1,
            "mAP": map_value,
            "ap_per_class": ap_per_class,
            "acc_per_class": acc_per_class,
        }

    def fit(self, train_loader, val_loader, epochs):
        best_map = -1.0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            metrics = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(metrics["val_loss"])
            self.history["exact_match_acc"].append(metrics["exact_match_acc"])
            self.history["label_acc"].append(metrics["label_acc"])
            self.history["f1_macro"].append(metrics["f1_macro"])
            self.history["mAP"].append(metrics["mAP"])

            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss       : {train_loss:.4f}")
            print(f"Val Loss         : {metrics['val_loss']:.4f}")
            print(f"Exact Match Acc  : {metrics['exact_match_acc']:.4f}")
            print(f"Label Accuracy   : {metrics['label_acc']:.4f}")
            print(f"F1 Macro         : {metrics['f1_macro']:.4f}")
            print(f"mAP              : {metrics['mAP']:.4f}")

            for i, class_name in enumerate(self.class_names):
                print(
                    f"{class_name}: "
                    f"Acc={metrics['acc_per_class'][i]:.4f}, "
                    f"AP={metrics['ap_per_class'][i]:.4f}"
                )

            if metrics["mAP"] > best_map:
                best_map = metrics["mAP"]
                save_checkpoint(self.model, os.path.join(self.output_dir, "best_model.pth"))
                print("[Checkpoint] Saved best_model.pth")

            if self.scheduler is not None:
                self.scheduler.step()

            with open(os.path.join(self.output_dir, "history.json"), "w") as f:
                json.dump(self.history, f, indent=2)

        save_checkpoint(self.model, os.path.join(self.output_dir, "last_model.pth"))
        print("[Checkpoint] Saved last_model.pth")

        return self.history