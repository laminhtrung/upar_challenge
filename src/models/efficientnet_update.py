import torch
import torch.nn as nn
from torchvision import models


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden_planes = max(in_planes // ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in [3, 7], "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientNetUpdateModel(nn.Module):
    def __init__(self, num_classes=40, pretrained=True, dropout=0.3):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)

        self.features = base_model.features
        self.attention = CBAM(channels=1280, ratio=16, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)          # [B, 1280, H', W']
        x = self.attention(x)
        x = self.pool(x)              # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)       # [B, 1280]
        logits = self.head(x)
        return logits