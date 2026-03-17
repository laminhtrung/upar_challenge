import torch
import torch.nn as nn
from torchvision import models


class ResNetAttrModel(nn.Module):
    def __init__(self, num_classes=40, pretrained=True, dropout=0.5):
        super().__init__()

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits