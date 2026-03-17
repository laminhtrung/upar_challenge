import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


CLASS_GROUPS = {
    "age": (0, 3),            # multiclass
    "gender": (3, 4),         # binary
    "hair": (4, 7),           # multiclass
    "ub_len": (7, 8),         # binary
    "ub_color": (8, 20),      # multiclass
    "lb_len": (20, 21),       # binary
    "lb_color": (21, 33),     # multiclass
    "lb_type": (33, 35),      # multiclass
    "backpack": (35, 36),     # binary
    "bag": (36, 37),          # binary
    "glasses": (37, 39),      # multiclass
    "hat": (39, 40),          # binary
}


def postprocess_predictions(logits):
    probs = torch.sigmoid(logits)
    preds = torch.zeros_like(probs)

    batch_idx = torch.arange(probs.shape[0], device=probs.device)

    # multiclass groups -> argmax
    for key in ["age", "hair", "ub_color", "lb_color", "lb_type", "glasses"]:
        s, e = CLASS_GROUPS[key]
        group_probs = probs[:, s:e]
        max_idx = group_probs.argmax(dim=1)
        preds[batch_idx, s + max_idx] = 1.0

    # binary groups -> threshold 0.5
    for key in ["gender", "ub_len", "lb_len", "backpack", "bag", "hat"]:
        s, e = CLASS_GROUPS[key]
        preds[:, s:e] = (probs[:, s:e] > 0.5).float()

    return probs, preds


def calculate_map(probs, labels, num_classes):
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    ap_per_class = []
    for class_idx in range(num_classes):
        try:
            ap = average_precision_score(labels[:, class_idx], probs[:, class_idx])
        except ValueError:
            ap = 0.0
        ap_per_class.append(ap)

    return float(np.mean(ap_per_class)), ap_per_class


def calculate_f1(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return f1_score(labels, preds, average="macro", zero_division=0)


def exact_match_accuracy(preds, labels):
    return (preds == labels).all(dim=1).float().mean().item()


def label_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def per_class_accuracy(preds, labels):
    correct = (preds == labels).float().sum(dim=0)
    total = labels.shape[0]
    return (correct / total).detach().cpu().tolist()