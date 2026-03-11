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
# PHYSICS MODULE 1: TOPO-AWARE WIND ADVECTION
# Deformable conv with offsets from wind + topography.
# The topo channel teaches the network that mountains BLOCK wind.
# ============================================================

class WindAdvectionBlock(nn.Module):
    """
    Deformable conv whose sampling offsets come from:
      - 16h of u10 + 16h of v10 = 32 wind channels (full trajectory)
      - 1 topography channel (orographic blocking)

    Over flat terrain: offsets follow wind direction freely.
    Near Himalayas: topo suppresses offsets → pollution pools at mountain base.
    """
    def __init__(self, channels, wind_channels=33):
        super().__init__()
        mid = max(channels // 4, 32)

        self.compress = nn.Conv2d(channels, mid, 1)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(wind_channels, 32, kernel_size=3, padding=1),  # 33 = 32 wind + 1 topo
            nn.GELU(),
            nn.Conv2d(32, 2 * 3 * 3, kernel_size=3, padding=1)
        )
        self.deform_weight = nn.Parameter(torch.randn(mid, mid, 3, 3) * 0.02)
        self.deform_bias = nn.Parameter(torch.zeros(mid))
        self.expand = nn.Conv2d(mid, channels, 1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, features, wind_topo):
        x = self.compress(features)
        offsets = self.offset_conv(wind_topo)
        x = deform_conv2d(x, offsets, self.deform_weight,
                          bias=self.deform_bias, padding=1)
        x = self.expand(x)
        return features + F.gelu(self.norm(x))


# ============================================================
# PHYSICS MODULE 2: EMISSION SOURCE INJECTION (Bottleneck)
# Static emissions → additive source bias at the bottleneck.
# ============================================================

class EmissionSourceLayer(nn.Module):
    """Injects emission sources (static, time-invariant) at the bottleneck."""
    def __init__(self, channels):
        super().__init__()
        self.source_bias = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, channels, kernel_size=1)
        )

    def forward(self, features, emissions):
        return features + F.relu(self.source_bias(emissions))


# ============================================================
# PHYSICS MODULE 3: HOURLY PHYSICS GATE (Output)
# Per-hour FiLM: applies hour-specific PBLH/Rain to each prediction.
# Solves the "Nighttime Trap" and "Time-Traveling Rain" edge cases.
# ============================================================

class HourlyPhysicsGate(nn.Module):
    """
    Applies weather-specific modulation to EACH forecast hour individually.
    Input: PBLH_h + Rain_h → sigmoid gate that dilutes/washes out that hour.

    Applied at the OUTPUT (not bottleneck) so Hour 1 rain only affects Hour 1.
    Processes all 16 hours in parallel via batch reshaping (no Python loop).
    """
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, pblh_all, rain_all):
        """
        Args:
            pblh_all: (B, 16, H, W) — per-hour PBLH
            rain_all: (B, 16, H, W) — per-hour rain mask
        Returns:
            modifier: (B, 16, H, W) — per-hour multiplicative gate
        """
        b, t, h, w = pblh_all.shape
        # Batch all 16 hours together: (B*16, 2, H, W)
        pblh_flat = pblh_all.reshape(b * t, 1, h, w)
        rain_flat = rain_all.reshape(b * t, 1, h, w)
        gate_in = torch.cat([pblh_flat, rain_flat], dim=1)
        modifier = self.gate(gate_in)                      # (B*16, 1, H, W)
        return modifier.reshape(b, t, h, w)                # (B, 16, H, W)


# ============================================================
# PHYSICS MODULE 4: RESIDUAL ADVECTION
# Physically shifts last_pm25 by wind (topo-aware) for each hour.
# Solves the "Ghost Plume" problem — the residual baseline MOVES.
# ============================================================

class ResidualAdvection(nn.Module):
    """
    Creates a wind-shifted version of last_pm25 as the residual baseline.
    The U-Net then predicts the DELTA from this moving baseline, not a static one.
    Uses topo-aware offsets so the shifted baseline respects mountains.
    """
    def __init__(self):
        super().__init__()
        # wind(2) + topo(1) → offsets for a 3x3 deformable identity conv
        self.offset_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2 * 3 * 3, kernel_size=3, padding=1)
        )
        # Initialize as identity (Dirac: center pixel = 1)
        self.weight = nn.Parameter(torch.zeros(1, 1, 3, 3))
        nn.init.dirac_(self.weight)

    def forward(self, pm25, avg_wind, topo):
        """
        Args:
            pm25: (B, 1, H, W)
            avg_wind: (B, 2, H, W)
            topo: (B, 1, H, W)
        Returns:
            shifted_pm25: (B, 1, H, W)
        """
        offset_input = torch.cat([avg_wind, topo], dim=1)
        offsets = self.offset_conv(offset_input)
        return deform_conv2d(pm25, offsets, self.weight, padding=1)


# ============================================================
# WIND-DEFORMABLE U-NET v2 (WD-UNet v2)
# Class name kept as FNO2D for backward-compatible imports.
#
# Key differences from v1:
#   1. Topo-aware wind offsets (Himalayan blocking)
#   2. Per-hour PBLH/Rain gate at OUTPUT (not bottleneck)
#   3. Advected residual (moving baseline, not static)
#   4. Emissions-only source injection at bottleneck
# ============================================================

class FNO2D(nn.Module):
    """
    WD-UNet v2: Output Physics Model.

    The U-Net predicts a 16-hour "Base Chemical Plume."
    Physics is applied at the OUTPUT per-hour, not at the bottleneck with averages.

    Input: (B, H, W, 278), Output: (B, H, W, 16)
    """

    _F_U10 = 2
    _F_V10 = 3
    _F_PBLH = 5
    _F_RAIN_MASK = 9

    def __init__(self, in_channels, time_out=16, width=128, modes=None, time_input=10):
        super().__init__()
        self.time_out = time_out
        self.time_input = time_input
        self.num_temporal_feats = (in_channels - time_input - 7 - 1) // 26

        c0 = width // 2      # 64
        c1 = width            # 128
        c2 = width * 2        # 256

        # --- Encoder ---
        self.enc0 = nn.Sequential(ConvBlock(in_channels + 2, c0), ConvBlock(c0, c0))
        self.enc1 = nn.Sequential(ConvBlock(c0, c1, stride=2), ConvBlock(c1, c1))
        self.enc2 = nn.Sequential(ConvBlock(c1, c2, stride=2), ConvBlock(c2, c2))

        # --- Bottleneck Physics ---
        self.wind_advection = WindAdvectionBlock(c2, wind_channels=33)   # 32 wind + 1 topo
        self.emission_source = EmissionSourceLayer(c2)
        self.bottleneck = nn.Sequential(ConvBlock(c2, c2), nn.Dropout(p=0.1))

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(c1 + c1, c1), ConvBlock(c1, c1))
        self.up0 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(ConvBlock(c0 + c0, c0), ConvBlock(c0, c0))

        # --- Output Head ---
        self.output_head = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=1), nn.GELU(),
            nn.Conv2d(c0, time_out, kernel_size=1)
        )

        # --- Output Physics (per-hour) ---
        self.hourly_gate = HourlyPhysicsGate()
        self.residual_advection = ResidualAdvection()

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device).view(1, 1, nx, 1).expand(b, 1, nx, ny)
        gridy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, ny).expand(b, 1, nx, ny)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        b, nx, ny, _ = x.shape
        nf = self.num_temporal_feats

        # ========================================
        # EXTRACT ALL PHYSICS FEATURES
        # ========================================

        last_pm25 = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)  # (B,1,H,W)

        temporal = x[..., self.time_input:self.time_input + nf * 26]
        temporal = temporal.reshape(b, nx, ny, 26, nf)
        future = temporal[:, :, :, self.time_input:, :]  # (B, H, W, 16, nf)

        # Stacked wind: all 16 future hours of u10,v10 → (B, 32, H, W)
        future_u = future[..., self._F_U10].permute(0, 3, 1, 2)       # (B, 16, H, W)
        future_v = future[..., self._F_V10].permute(0, 3, 1, 2)       # (B, 16, H, W)
        stacked_wind = torch.cat([future_u, future_v], dim=1)          # (B, 32, H, W)

        # Average wind for residual advection
        avg_wind = torch.cat([
            future_u.mean(dim=1, keepdim=True),
            future_v.mean(dim=1, keepdim=True)
        ], dim=1)                                                       # (B, 2, H, W)

        # Per-hour PBLH and Rain → (B, 16, H, W) each
        future_pblh = future[..., self._F_PBLH].permute(0, 3, 1, 2)   # (B, 16, H, W)
        future_rain = future[..., self._F_RAIN_MASK].permute(0, 3, 1, 2)

        # Static features
        emissions = x[..., -8:-1].permute(0, 3, 1, 2)                 # (B, 7, H, W)
        topo = x[..., -1:].permute(0, 3, 1, 2)                        # (B, 1, H, W)

        # ========================================
        # ENCODER
        # ========================================
        x_in = torch.cat([x.permute(0, 3, 1, 2),
                          self.get_grid(b, nx, ny, x.device)], dim=1)

        e0 = self.enc0(x_in)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)

        # ========================================
        # BOTTLENECK: TOPO-AWARE ADVECTION + EMISSIONS
        # ========================================
        bn_size = e2.shape[2:]
        wind_bn = F.interpolate(stacked_wind, size=bn_size, mode='bilinear', align_corners=False)
        topo_bn = F.interpolate(topo, size=bn_size, mode='bilinear', align_corners=False)
        emi_bn = F.interpolate(emissions, size=bn_size, mode='bilinear', align_corners=False)

        # Wind + Topo → topo-aware offsets (Himalayan blocking)
        wind_topo_bn = torch.cat([wind_bn, topo_bn], dim=1)           # (B, 33, bnH, bnW)
        feat = self.wind_advection(e2, wind_topo_bn)
        feat = self.emission_source(feat, emi_bn)
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
        # OUTPUT: BASE FORECAST + PHYSICS GATES
        # ========================================

        # 1. Base 16-hour forecast from U-Net
        base = self.output_head(d0)                                    # (B, 16, H, W)

        # 2. ADVECTED RESIDUAL: shift last_pm25 by wind (not static!)
        shifted_pm25 = self.residual_advection(last_pm25, avg_wind, topo)  # (B, 1, H, W)
        base = base + shifted_pm25                                     # broadcast over 16 hours

        # 3. PER-HOUR PHYSICS GATE: hour-specific PBLH/Rain modulation
        modifier = self.hourly_gate(future_pblh, future_rain)          # (B, 16, H, W)
        out = base * modifier

        return out.permute(0, 2, 3, 1)                                 # (B, H, W, 16)