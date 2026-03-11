import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


# ============================================================
# BUILDING BLOCK: Conv → GroupNorm → GELU
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(4, out_ch)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


# ============================================================
# PHYSICS MODULE 1: WIND-GUIDED DEFORMABLE CONVOLUTION
# Replaces rigid PDE advection with learned, wind-shifted receptive fields.
# Based on "Deformable Convolutional Networks" (Zhu et al., ICCV 2017)
# ============================================================

class WindAdvectionBlock(nn.Module):
    """
    Deformable convolution whose sampling offsets are generated from wind vectors.
    The conv kernel physically shifts UPWIND to gather incoming pollution.
    Uses channel compression for parameter efficiency.
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 4, 32)

        self.compress = nn.Conv2d(channels, mid, 1)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2 * 3 * 3, kernel_size=3, padding=1)   # 18 offsets per pixel
        )
        self.deform_weight = nn.Parameter(torch.randn(mid, mid, 3, 3) * 0.02)
        self.deform_bias = nn.Parameter(torch.zeros(mid))
        self.expand = nn.Conv2d(mid, channels, 1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, features, wind_uv):
        x = self.compress(features)
        offsets = self.offset_conv(wind_uv)
        x = deform_conv2d(x, offsets, self.deform_weight,
                          bias=self.deform_bias, padding=1)
        x = self.expand(x)
        return features + F.gelu(self.norm(x))       # residual


# ============================================================
# PHYSICS MODULE 2: FiLM CONDITIONING (Feature-wise Linear Modulation)
# Replaces rigid PDE sinks/sources with learned multiplicative & additive gates.
# Based on "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
# ============================================================

class PhysicsModulationLayer(nn.Module):
    """
    FiLM-style physics conditioning:
        New_Features = Features * sigmoid(f(PBLH, Rain)) + ReLU(g(Emissions))

    - PBLH + Rain → multiplicative gate (dilution / wet scavenging)
    - Emissions   → additive bias (pollution sources)
    """
    def __init__(self, channels):
        super().__init__()
        self.sink_gate = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=1)
        )
        self.source_bias = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=1)
        )

    def forward(self, features, pblh_rain, emissions):
        dilution = torch.sigmoid(self.sink_gate(pblh_rain))
        sources = F.relu(self.source_bias(emissions))
        return features * dilution + sources


# ============================================================
# WIND-DEFORMABLE U-NET (WD-UNet)
# Class name kept as FNO2D for backward-compatible imports.
#
# Direct multi-step (no RNN loop):
#   Encoder → [WindAdvection + FiLM at bottleneck] → Decoder
#   Skip connections preserve sharp PM2.5 hotspot boundaries.
#   Single-shot 16-hour prediction.
# ============================================================

class FNO2D(nn.Module):
    """
    Wind-Deformable U-Net for PM2.5 forecasting.

    Input layout (B, H, W, 278):
        [0:10]    PM2.5 history (10 hours)
        [10:270]  Temporal (10 features × 26 hours), ch = 10 + t*10 + f
        [270:277] Static emissions (7 channels)
        [277]     Topography (1 channel)

    Output: (B, H, W, 16) — PM2.5 predictions for 16 future hours
    """

    # Feature indices within the 10 temporal features per timestep
    _F_U10 = 2
    _F_V10 = 3
    _F_PBLH = 5
    _F_RAIN_MASK = 9

    def __init__(self, in_channels, time_out=16, width=128, modes=None, time_input=10):
        super().__init__()
        self.time_out = time_out
        self.time_input = time_input
        self.num_temporal_feats = (in_channels - time_input - 7 - 1) // 26

        # U-Net channel pyramid (controlled by the width config param)
        c0 = width // 2     # 64
        c1 = width           # 128
        c2 = width * 2       # 256

        # --- Encoder ---
        self.enc0 = nn.Sequential(ConvBlock(in_channels + 2, c0), ConvBlock(c0, c0))
        self.enc1 = nn.Sequential(ConvBlock(c0, c1, stride=2), ConvBlock(c1, c1))
        self.enc2 = nn.Sequential(ConvBlock(c1, c2, stride=2), ConvBlock(c2, c2))

        # --- Physics at Bottleneck ---
        self.wind_advection = WindAdvectionBlock(c2)
        self.physics_mod = PhysicsModulationLayer(c2)
        self.bottleneck = nn.Sequential(
            ConvBlock(c2, c2),
            nn.Dropout(p=0.1)
        )

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(c1 + c1, c1), ConvBlock(c1, c1))

        self.up0 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(ConvBlock(c0 + c0, c0), ConvBlock(c0, c0))

        # --- Output Head ---
        self.output_head = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c0, time_out, kernel_size=1)
        )

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device).view(1, 1, nx, 1).expand(b, 1, nx, ny)
        gridy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, ny).expand(b, 1, nx, ny)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, 278)
        Returns:
            (B, H, W, 16) PM2.5 predictions
        """
        b, nx, ny, _ = x.shape
        nf = self.num_temporal_feats

        # ========================================
        # EXTRACT PHYSICS FEATURES
        # ========================================

        # Last known PM2.5 (for residual connection)
        last_pm25 = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)  # (B,1,H,W)

        # Unflatten temporal → (B, H, W, 26, nf)
        temporal = x[..., self.time_input:self.time_input + nf * 26]
        temporal = temporal.reshape(b, nx, ny, 26, nf)
        future = temporal[:, :, :, self.time_input:, :]   # (B, H, W, 16, nf)

        # Average future wind → (B, 2, H, W) for deformable conv
        avg_u = future[..., self._F_U10].mean(dim=3).unsqueeze(1)    # (B,1,H,W)
        avg_v = future[..., self._F_V10].mean(dim=3).unsqueeze(1)    # (B,1,H,W)
        avg_wind = torch.cat([avg_u, avg_v], dim=1)

        # Average future PBLH + Rain → (B, 2, H, W) for FiLM
        avg_pblh = future[..., self._F_PBLH].mean(dim=3).unsqueeze(1)
        avg_rain = future[..., self._F_RAIN_MASK].mean(dim=3).unsqueeze(1)
        pblh_rain = torch.cat([avg_pblh, avg_rain], dim=1)

        # Static emissions → (B, 7, H, W)
        emissions = x[..., -8:-1].permute(0, 3, 1, 2)

        # ========================================
        # ENCODER
        # ========================================
        x_in = torch.cat([x.permute(0, 3, 1, 2),
                          self.get_grid(b, nx, ny, x.device)], dim=1)  # (B, 280, H, W)

        e0 = self.enc0(x_in)     # (B, c0, 140, 124)
        e1 = self.enc1(e0)       # (B, c1, 70, 62)
        e2 = self.enc2(e1)       # (B, c2, 35, 31)

        # ========================================
        # PHYSICS AT BOTTLENECK
        # ========================================
        bn_size = e2.shape[2:]
        wind_bn = F.interpolate(avg_wind, size=bn_size, mode='bilinear', align_corners=False)
        pr_bn = F.interpolate(pblh_rain, size=bn_size, mode='bilinear', align_corners=False)
        emi_bn = F.interpolate(emissions, size=bn_size, mode='bilinear', align_corners=False)

        feat = self.wind_advection(e2, wind_bn)       # advection
        feat = self.physics_mod(feat, pr_bn, emi_bn)   # dilution + sources
        feat = self.bottleneck(feat)

        # ========================================
        # DECODER WITH SKIP CONNECTIONS
        # ========================================
        d1 = self.up1(feat)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.up0(d1)
        d0 = F.interpolate(d0, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        # ========================================
        # OUTPUT + RESIDUAL
        # ========================================
        out = self.output_head(d0)       # (B, 16, H, W)
        out = out + last_pm25            # broadcast residual

        return out.permute(0, 2, 3, 1)   # (B, H, W, 16)