"""
Geometric Co-Evolution Segmentation Model with Neural ODE and Expert Mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights
from torchdiffeq import odeint_adjoint as odeint

# Optional: For transformer-based backbones in future
# from transformers import SegformerModel

class ExpertModule(nn.Module):
    """
    Expert module for a specific class or class group.
    Supports shared information utilization from other experts.
    """
    def __init__(self, in_channels, out_channels=256, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        # Gating mechanism for shared features
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, shared_features=None):
        """
        :param x: Input features from backbone
        :param shared_features: List of features from other experts (optional)
        :return: Segmentation logit and gate-weighted features
        """
        out = self.relu(self.bn1(self.conv1(x)))
        if shared_features is not None:
            # Apply soft gate to shared features
            for feat in shared_features:
                gate = self.gate(feat)
                out = out + gate * feat
        logit = self.classifier(out)
        return logit, out


class LatentOutputODE(nn.Module):
    """
    Neural ODE function for joint evolution of latent features and logits.
    F_psi(h_t, t) where h_t = [z_t, y_t]
    """
    def __init__(self, in_channels, num_classes, hidden_dim=64):
        super().__init__()
        self.conv_z = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(num_classes, hidden_dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_dim, in_channels + num_classes, kernel_size=3, padding=1)
        self.swish = nn.SiLU()

        # Time embedding: sin(2πt/T_stage + φ)
        self.T_stage = 50
        self.register_buffer('phi', torch.rand(1) * 2 * 3.14159)

    def forward(self, t, h):
        """
        h: [B, C+K, H, W] where C = feature dim, K = num_classes
        """
        z, y = h[:, :-y.shape[1]], h[:, -y.shape[1]:]
        t_pe = torch.sin(2 * 3.14159 / self.T_stage * t + self.phi).view(1, -1, 1, 1)
        t_pe = t_pe.expand(z.size(0), -1, z.size(2), z.size(3))

        # Concat time PE to z
        z_t = torch.cat([z, t_pe], dim=1)
        out = self.swish(self.conv_z(z_t) + self.conv_y(y))
        dh = self.conv_out(out)
        return dh


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module (DeepLabv3+).
    """
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(len(rates) * out_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        global_feat = F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        local_feats = [conv(x) for conv in self.convs]
        out = torch.cat([global_feat] + local_feats, dim=1)
        return self.project(out)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Head with encoder-decoder structure.
    """
    def __init__(self, low_level_channels, num_classes):
        super().__init__()
        self.low_proj = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.low_bn = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.aspp = ASPP(2048, 256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.relu(self.low_bn(self.low_proj(low_level_feat)))
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_level_feat], dim=1)
        return self.decoder(x)


class GCoeSegmentationModel(nn.Module):
    """
    Main Segmentation Model with:
    - ResNet101 Backbone
    - DeepLabV3+ Head
    - Expert Modules per Class Group
    - Neural ODE for Latent-Output Co-Evolution
    """
    def __init__(self, num_classes, num_experts=6, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts

        # Backbone (Encoder)
        if pretrained:
            weights = ResNet101_Weights.IMAGENET1K_V2
        else:
            weights = None
        backbone = resnet101(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # Low-level features (256 channels)
            backbone.layer2,  # (512)
            backbone.layer3,  # (1024)
            backbone.layer4   # High-level features (2048)
        )
        self.low_level_channels = 256  # From layer1

        # DeepLab Head
        self.head = DeepLabV3Plus(low_level_channels=256, num_classes=num_classes)

        # Expert Modules
        self.experts = nn.ModuleList([
            ExpertModule(in_channels=256, num_classes=num_classes // num_experts + 1)
            for _ in range(num_experts)
        ])

        # Neural ODE for co-evolution
        self.use_ode = True
        self.ode_func = LatentOutputODE(in_channels=256, num_classes=num_classes)
        self.t_span = torch.linspace(0, 1, 10)  # 10 integration steps

        # Freeze BN in backbone (common in CSS)
        self._freeze_bn()

    def _freeze_bn(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x, stage_idx=None, use_ode=True):
        """
        :param x: Input image tensor
        :param stage_idx: Current stage index (for expert routing)
        :param use_ode: Whether to apply Neural ODE co-evolution
        :return: Segmentation logits
        """
        h, w = x.shape[-2:]

        # Extract features
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 4:  # After layer1 (low-level)
                low_level_feat = x

        # DeepLab Head
        logits = self.head(x, low_level_feat)

        # Expert processing (optional, for class-balanced replay)
        if stage_idx is not None and self.experts:
            expert_idx = stage_idx % len(self.experts)
            shared_feats = []
            for i, expert in enumerate(self.experts):
                if i != expert_idx:
                    _, feat = expert(low_level_feat)
                    shared_feats.append(feat)
            logits_expert, _ = self.experts[expert_idx](low_level_feat, shared_feats)
            logits = logits + 0.5 * logits_expert  # Fuse

        # Apply Neural ODE for co-evolution of z and y
        if use_ode and self.use_ode:
            z = F.adaptive_avg_pool2d(x, (16, 16))  # Downsampled latent
            y = F.interpolate(logits, size=(16, 16), mode='bilinear', align_corners=False)
            h0 = torch.cat([z, y], dim=1)  # [B, C+K, 16, 16]

            h_final = odeint(
                self.ode_func,
                h0,
                self.t_span.to(x.device),
                method='dopri5',
                atol=1e-5,
                rtol=1e-4
            )[-1]  # Final state

            # Extract evolved logits
            _, y_final = torch.split(h_final, [z.shape[1], y.shape[1]], dim=1)
            logits = F.interpolate(y_final, size=(h, w), mode='bilinear', align_corners=False)

        return logits

    def get_backbone_features(self, x):
        """Extract backbone features for memory or distillation."""
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i == 7:  # After layer4
                break
        return x


# __init__.py will import this
__all__ = ['GCoeSegmentationModel', 'ExpertModule', 'LatentOutputODE', 'DeepLabV3Plus']