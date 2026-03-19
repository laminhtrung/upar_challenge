import torch
import torch.nn as nn
import timm


class PCBRealAttrModel(nn.Module):
    def __init__(
        self,
        num_classes=40,
        pretrained=True,
        backbone_name="resnet50",
        num_parts=6,
        reduced_dim=256,
        dropout=0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parts = num_parts
        self.backbone_name = backbone_name

        # Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        feat_dim = self.backbone.num_features

        # Pool thành num_parts phần
        self.part_pool = nn.AdaptiveAvgPool2d((num_parts, 1))

        # Giảm chiều
        self.reduction = nn.Sequential(
            nn.Conv2d(feat_dim, reduced_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_dim),
            nn.ReLU(inplace=True)
        )

        # Classifier từng part
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(reduced_dim, num_classes)
            )
            for _ in range(num_parts)
        ])

    def forward(self, x):
        feat = self.backbone.forward_features(x)

        if isinstance(feat, (list, tuple)):
            feat = feat[-1]

        if feat.dim() != 4:
            raise ValueError(
                f"Backbone '{self.backbone_name}' phải trả về (B,C,H,W), "
                f"nhưng nhận được {tuple(feat.shape)}"
            )

        part_feat = self.part_pool(feat)
        part_feat = self.reduction(part_feat)

        logits_list = []

        for i in range(self.num_parts):
            part = part_feat[:, :, i, :].squeeze(-1)
            logits = self.classifiers[i](part)
            logits_list.append(logits)

        logits = torch.stack(logits_list, dim=0)
        logits = logits.mean(dim=0)

        return logits