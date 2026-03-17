import torch.nn as nn
import timm


class ViTAttrModel(nn.Module):
    def __init__(self, num_classes=40, pretrained=True, dropout=0.1):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.model(x)