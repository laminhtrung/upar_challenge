import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HybridEfficientNetTransformerV3(nn.Module):
    """
    Improved hybrid model for pedestrian attribute recognition.

    Architecture:
        Input
          -> EfficientNet-B0 multi-scale features
          -> channel projection for each scale
          -> upsample and fuse
          -> positional depthwise conv
          -> Transformer encoder
          -> attention pooling (transformer global)
          -> CNN global pooling
          -> part-based pooling (vertical body partitions)
          -> fused classifier

    Main improvements:
        1. Multi-scale feature fusion
        2. Positional encoding by depthwise convolution
        3. Attention pooling instead of mean pooling
        4. CNN residual global branch
        5. Part-based body pooling for attribute recognition
    """

    def __init__(
        self,
        num_classes: int = 40,
        pretrained: bool = True,
        dropout: float = 0.4,
        backbone_name: str = "efficientnet_b0",
        d_model: int = 256,
        nhead: int = 8,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        num_parts: int = 3,
    ):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert num_parts >= 1, "num_parts must be >= 1"

        self.num_classes = num_classes
        self.d_model = d_model
        self.num_parts = num_parts

        # Multi-scale EfficientNet feature extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-2, -1],
        )

        feat_channels = self.backbone.feature_info.channels()
        c_low = feat_channels[-2]
        c_high = feat_channels[-1]

        # Project both scales to same dimension
        self.proj_low = nn.Conv2d(c_low, d_model, kernel_size=1, bias=False)
        self.proj_high = nn.Conv2d(c_high, d_model, kernel_size=1, bias=False)

        self.bn_low = nn.BatchNorm2d(d_model)
        self.bn_high = nn.BatchNorm2d(d_model)

        # Fuse low + high scale
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
        )

        # Positional information
        self.pos_conv = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=3,
            padding=1,
            groups=d_model,
            bias=False,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.token_norm = nn.LayerNorm(d_model)

        # Attention pooling over tokens
        attn_hidden = max(d_model // 2, 32)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, attn_hidden),
            nn.GELU(),
            nn.Linear(attn_hidden, 1),
        )

        # CNN global pooling
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Part-based projection
        self.part_proj = nn.Sequential(
            nn.Linear(d_model * num_parts, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final classifier
        fusion_dim = d_model * 3  # cnn_global + trans_global + part_feat
        hidden_dim = d_model * 2

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _extract_part_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Split feature map vertically into body parts and average-pool each part.

        Args:
            feat: Tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, C * num_parts]
        """
        b, c, h, w = feat.shape

        part_feats = []
        boundaries = torch.linspace(
            0, h, steps=self.num_parts + 1, device=feat.device
        ).long()

        for i in range(self.num_parts):
            start = int(boundaries[i].item())
            end = int(boundaries[i + 1].item())

            # safe guard for very small H
            if end <= start:
                end = min(start + 1, h)

            part = feat[:, :, start:end, :]
            pooled = part.mean(dim=(2, 3))  # [B, C]
            part_feats.append(pooled)

        return torch.cat(part_feats, dim=1)  # [B, C * num_parts]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Extract multi-scale features
        feats = self.backbone(x)
        feat_low = feats[-2]   # more local detail
        feat_high = feats[-1]  # more semantic/global

        # 2) Project channels
        feat_low = self.bn_low(self.proj_low(feat_low))
        feat_high = self.bn_high(self.proj_high(feat_high))

        # 3) Resize high-level feature to low-level resolution
        feat_high_up = F.interpolate(
            feat_high,
            size=feat_low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 4) Fuse multi-scale features
        feat = torch.cat([feat_low, feat_high_up], dim=1)
        feat = self.fuse_conv(feat)   # [B, d_model, H, W]

        # 5) Inject spatial/positional information
        feat = feat + self.pos_conv(feat)

        # 6) CNN global branch
        cnn_global = self.cnn_pool(feat).flatten(1)   # [B, d_model]

        # 7) Part-based branch
        part_feat = self._extract_part_features(feat)  # [B, d_model * num_parts]
        part_feat = self.part_proj(part_feat)          # [B, d_model]

        # 8) Transformer branch
        tokens = feat.flatten(2).transpose(1, 2)      # [B, H*W, d_model]
        tokens = self.transformer(tokens)             # [B, N, d_model]
        tokens = self.token_norm(tokens)

        attn = self.attn_pool(tokens)                 # [B, N, 1]
        attn = torch.softmax(attn, dim=1)
        trans_global = (tokens * attn).sum(dim=1)     # [B, d_model]

        # 9) Fuse final representations
        fused = torch.cat([cnn_global, trans_global, part_feat], dim=1)
        fused = self.dropout(fused)

        # 10) Classification
        logits = self.classifier(fused)
        return logits