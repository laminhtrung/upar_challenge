import torch.nn as nn
from torchvision import models


class EfficientNetAttrModel(nn.Module):
    def __init__(self, num_classes=40, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

        self.model = model

    def forward(self, x):
        return self.model(x)