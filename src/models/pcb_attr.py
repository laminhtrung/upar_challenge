import torch
import torch.nn as nn
from torchvision import models


class PCBAttrModel(nn.Module):
    def __init__(self, num_classes=40, pretrained=True, num_parts=6, dropout=0.5, reduced_dim=256):
        super().__init__()
        self.num_parts = num_parts

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # B,2048,H,W

        self.pool = nn.AdaptiveAvgPool2d((num_parts, 1))
        self.reduce = nn.Conv2d(2048, reduced_dim, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(reduced_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(reduced_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)                  # (B, 2048, H, W)
        x = self.pool(x)                      # (B, 2048, num_parts, 1)
        x = self.reduce(x)                    # (B, reduced_dim, num_parts, 1)
        x = x.squeeze(-1).permute(0, 2, 1)    # (B, num_parts, reduced_dim)

        x = torch.mean(x, dim=1)              # (B, reduced_dim)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.head(x)
        return logits