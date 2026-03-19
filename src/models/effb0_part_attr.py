import torch
import torch.nn as nn
import torchvision.models as models


class PartBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=256, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


class EfficientNetB0PartAttrModel(nn.Module):
    """
    EfficientNet-B0 + Global Branch + Part-based Local Branch
    for multi-label attribute recognition.

    Output:
        logits: [B, num_classes]
    """

    def __init__(
        self,
        num_classes=40,
        pretrained=True,
        num_parts=4,
        embed_dim=256,
        dropout=0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_parts = num_parts
        self.embed_dim = embed_dim

        # ---------------------------
        # Backbone: EfficientNet-B0
        # ---------------------------
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = models.efficientnet_b0(weights=weights)

        # chỉ lấy phần feature extractor
        self.backbone = backbone.features  # output: [B, 1280, H, W]
        self.out_channels = 1280

        # ---------------------------
        # Pooling
        # ---------------------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # chia thành num_parts phần theo chiều cao, mỗi part pool về 1 cột
        self.part_pool = nn.AdaptiveAvgPool2d((num_parts, 1))

        # ---------------------------
        # Global embedding
        # ---------------------------
        self.global_embed = PartBlock(
            in_channels=self.out_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # ---------------------------
        # Local part embeddings
        # ---------------------------
        self.part_embeds = nn.ModuleList([
            PartBlock(
                in_channels=self.out_channels,
                embed_dim=embed_dim,
                dropout=dropout
            )
            for _ in range(num_parts)
        ])

        # ---------------------------
        # Final classifier
        # concat: [global, part1, part2, ..., partN]
        # ---------------------------
        fused_dim = embed_dim * (num_parts + 1)

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        feat = self.backbone(x)  # [B, 1280, H, W]
        return feat

    def forward(self, x):
        feat = self.forward_features(x)  # [B, C, H, W]
        b, c, h, w = feat.shape

        # ---------------------------
        # Global branch
        # ---------------------------
        g = self.global_pool(feat).view(b, c)       # [B, C]
        g = self.global_embed(g)                    # [B, embed_dim]

        # ---------------------------
        # Part-based local branch
        # part_pool -> [B, C, num_parts, 1]
        # ---------------------------
        p = self.part_pool(feat).squeeze(-1)        # [B, C, num_parts]

        part_features = []
        for i in range(self.num_parts):
            pi = p[:, :, i]                         # [B, C]
            pi = self.part_embeds[i](pi)           # [B, embed_dim]
            part_features.append(pi)

        # concat global + all parts
        fused = torch.cat([g] + part_features, dim=1)   # [B, embed_dim*(num_parts+1)]

        logits = self.classifier(fused)                 # [B, num_classes]
        return logits