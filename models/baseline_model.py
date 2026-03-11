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
# PHYSICS: TOPO-AWARE WIND ADVECTION
# ============================================================

class WindAdvectionBlock(nn.Module):
    """Deformable conv: wind(32ch) + topo(1ch) → offsets."""
    def __init__(self, channels, wind_channels=33):
        super().__init__()
        mid = max(channels // 4, 32)
        self.compress = nn.Conv2d(channels, mid, 1)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(wind_channels, 32, kernel_size=3, padding=1),
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
# PHYSICS: EMISSION SOURCE INJECTION
# ============================================================

class EmissionSourceLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.source_bias = nn.Sequential(
            nn.Conv2d(7, 32, 1), nn.GELU(), nn.Conv2d(32, channels, 1)
        )

    def forward(self, features, emissions):
        return features + F.relu(self.source_bias(emissions))


# ============================================================
# PHYSICS: PER-HOUR PBLH/RAIN GATE
# ============================================================

class HourlyPhysicsGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

    def forward(self, pblh_all, rain_all):
        b, t, h, w = pblh_all.shape
        gate_in = torch.cat([
            pblh_all.reshape(b * t, 1, h, w),
            rain_all.reshape(b * t, 1, h, w)
        ], dim=1)
        return self.gate(gate_in).reshape(b, t, h, w)


# ============================================================
# PHYSICS: RESIDUAL ADVECTION
# ============================================================

class ResidualAdvection(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 2 * 3 * 3, 3, padding=1)
        )
        self.weight = nn.Parameter(torch.zeros(1, 1, 3, 3))
        nn.init.dirac_(self.weight)

    def forward(self, pm25, avg_wind, topo):
        offsets = self.offset_conv(torch.cat([avg_wind, topo], dim=1))
        return deform_conv2d(pm25, offsets, self.weight, padding=1)


# ============================================================
# WIND-DEFORMABLE U-NET (Clean WD-UNet v2)
# Class name: FNO2D for backward-compatible imports.
# ============================================================

class FNO2D(nn.Module):
    """
    Clean WD-UNet v2 with Output Physics.
    Log-transformed target makes the network predict in smooth log-space.

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

        c0 = width // 2
        c1 = width
        c2 = width * 2

        # Encoder
        self.enc0 = nn.Sequential(ConvBlock(in_channels + 2, c0), ConvBlock(c0, c0))
        self.enc1 = nn.Sequential(ConvBlock(c0, c1, stride=2), ConvBlock(c1, c1))
        self.enc2 = nn.Sequential(ConvBlock(c1, c2, stride=2), ConvBlock(c2, c2))

        # Bottleneck
        self.wind_advection = WindAdvectionBlock(c2, wind_channels=33)
        self.emission_source = EmissionSourceLayer(c2)
        self.bottleneck = nn.Sequential(ConvBlock(c2, c2), nn.Dropout(p=0.1))

        # Decoder
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(ConvBlock(c1 + c1, c1), ConvBlock(c1, c1))
        self.up0 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2)
        self.dec0 = nn.Sequential(ConvBlock(c0 + c0, c0), ConvBlock(c0, c0))

        # Output
        self.output_head = nn.Sequential(
            nn.Conv2d(c0, c0, 1), nn.GELU(), nn.Conv2d(c0, time_out, 1)
        )

        self.hourly_gate = HourlyPhysicsGate()
        self.residual_advection = ResidualAdvection()

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device).view(1, 1, nx, 1).expand(b, 1, nx, ny)
        gridy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, ny).expand(b, 1, nx, ny)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        b, nx, ny, _ = x.shape
        nf = self.num_temporal_feats

        # Extract features
        last_pm25 = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)

        temporal = x[..., self.time_input:self.time_input + nf * 26]
        temporal = temporal.reshape(b, nx, ny, 26, nf)
        future = temporal[:, :, :, self.time_input:, :]

        future_u = future[..., self._F_U10].permute(0, 3, 1, 2)
        future_v = future[..., self._F_V10].permute(0, 3, 1, 2)
        stacked_wind = torch.cat([future_u, future_v], dim=1)

        avg_wind = torch.cat([
            future_u.mean(dim=1, keepdim=True),
            future_v.mean(dim=1, keepdim=True)
        ], dim=1)

        future_pblh = future[..., self._F_PBLH].permute(0, 3, 1, 2)
        future_rain = future[..., self._F_RAIN_MASK].permute(0, 3, 1, 2)

        emissions = x[..., -8:-1].permute(0, 3, 1, 2)
        topo = x[..., -1:].permute(0, 3, 1, 2)

        # Encoder
        x_in = torch.cat([x.permute(0, 3, 1, 2),
                          self.get_grid(b, nx, ny, x.device)], dim=1)
        e0 = self.enc0(x_in)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)

        # Bottleneck
        bn_size = e2.shape[2:]
        wind_bn = F.interpolate(stacked_wind, size=bn_size, mode='bilinear', align_corners=False)
        topo_bn = F.interpolate(topo, size=bn_size, mode='bilinear', align_corners=False)
        emi_bn = F.interpolate(emissions, size=bn_size, mode='bilinear', align_corners=False)

        feat = self.wind_advection(e2, torch.cat([wind_bn, topo_bn], dim=1))
        feat = self.emission_source(feat, emi_bn)
        feat = self.bottleneck(feat)

        # Decoder
        d1 = self.up1(feat)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        d0 = self.up0(d1)
        d0 = F.interpolate(d0, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([d0, e0], dim=1))

        # Output + physics
        base = self.output_head(d0)
        shifted_pm25 = self.residual_advection(last_pm25, avg_wind, topo)
        base = base + shifted_pm25
        modifier = self.hourly_gate(future_pblh, future_rain)
        out = base * modifier

        return out.permute(0, 2, 3, 1)