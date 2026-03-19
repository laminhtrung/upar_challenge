import torch
import torch.nn as nn
import timm


class HybridEfficientNetTransformer(nn.Module):
    """
    EfficientNet-B0 backbone + lightweight Transformer encoder head.

    Flow:
        image -> EfficientNet feature map -> 1x1 projection -> tokens
              -> TransformerEncoder -> mean pooling -> dropout -> classifier
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
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],
        )

        backbone_out_channels = self.backbone.feature_info.channels()[-1]

        self.proj = nn.Conv2d(backbone_out_channels, d_model, kernel_size=1)

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

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # list of feature maps, take the last one
        feat = self.backbone(x)[-1]            # [B, C, H, W]
        feat = self.proj(feat)                 # [B, d_model, H, W]

        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # [B, H*W, d_model]

        tokens = self.transformer(tokens)          # [B, H*W, d_model]
        tokens = self.norm(tokens)

        pooled = tokens.mean(dim=1)                # [B, d_model]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)           # [B, num_classes]
        return logits