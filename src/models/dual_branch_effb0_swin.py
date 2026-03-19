import timm
import torch
import torch.nn as nn


class DualBranchEfficientNetSwin(nn.Module):
    def __init__(
        self,
        num_classes,
        pretrained=True,
        dropout=0.4,
        cnn_name="efficientnet_b0",
        transformer_name="swin_tiny_patch4_window7_224",
        fusion_dim=256,
        image_size=(384, 128),
    ):
        super().__init__()

        self.cnn_branch = timm.create_model(
            cnn_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.transformer_branch = timm.create_model(
            transformer_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,   # <-- thêm dòng này
        )

        cnn_dim = self.cnn_branch.num_features
        trans_dim = self.transformer_branch.num_features

        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.trans_proj = nn.Linear(trans_dim, fusion_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim * 2, num_classes)

    def forward(self, x):
        cnn_feat = self.cnn_branch(x)
        transformer_feat = self.transformer_branch(x)

        cnn_feat = self.cnn_proj(cnn_feat)
        transformer_feat = self.trans_proj(transformer_feat)

        fused = torch.cat([cnn_feat, transformer_feat], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)