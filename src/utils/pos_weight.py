import torch

def compute_pos_weight(train_loader, num_classes, device, max_value=10.0):
    pos_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    total_samples = 0

    for _, labels in train_loader:
        labels = labels.to(device).float()
        pos_counts += labels.sum(dim=0)
        total_samples += labels.size(0)

    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = torch.sqrt(pos_weight)

    # 🔥 thêm dòng này
    pos_weight = torch.clamp(pos_weight, max=max_value)

    return pos_weight