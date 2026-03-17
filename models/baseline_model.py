import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWavelet2D(nn.Module):
    """Native PyTorch 2D Haar Discrete Wavelet Transform."""
    def __init__(self, in_channels):
        super().__init__()
        h0 = [1 / 2,  1 / 2]
        h1 = [-1 / 2, 1 / 2]
        ll = torch.tensor([[h0[0]*h0[0], h0[0]*h0[1]], [h0[1]*h0[0], h0[1]*h0[1]]])
        lh = torch.tensor([[h1[0]*h0[0], h1[0]*h0[1]], [h1[1]*h0[0], h1[1]*h0[1]]])
        hl = torch.tensor([[h0[0]*h1[0], h0[0]*h1[1]], [h0[1]*h1[0], h0[1]*h1[1]]])
        hh = torch.tensor([[h1[0]*h1[0], h1[0]*h1[1]], [h1[1]*h1[0], h1[1]*h1[1]]])
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('filters', filters)
        self.in_channels = in_channels

    def forward(self, x):
        return F.conv2d(x, self.filters, stride=2, groups=self.in_channels)


class InverseHaarWavelet2D(nn.Module):
    """Native PyTorch 2D Inverse Haar Wavelet Transform."""
    def __init__(self, in_channels):
        super().__init__()
        h0 = [1.0,  1.0]
        h1 = [-1.0, 1.0]
        ll = torch.tensor([[h0[0]*h0[0], h0[0]*h0[1]], [h0[1]*h0[0], h0[1]*h0[1]]])
        lh = torch.tensor([[h1[0]*h0[0], h1[0]*h0[1]], [h1[1]*h0[0], h1[1]*h0[1]]])
        hl = torch.tensor([[h0[0]*h1[0], h0[0]*h1[1]], [h0[1]*h1[0], h0[1]*h1[1]]])
        hh = torch.tensor([[h1[0]*h1[0], h1[0]*h1[1]], [h1[1]*h1[0], h1[1]*h1[1]]])
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('filters', filters)
        self.in_channels = in_channels

    def forward(self, x):
        return F.conv_transpose2d(x, self.filters, stride=2, groups=self.in_channels)


class TemporalEncoder(nn.Module):
    """
    Compresses (num_temporal_features, total_time) per pixel into a fixed-size embedding.

    FIX (Bug #3): The original used flatten + Linear(hidden*total_time, out), creating
    a (B*H*W, hidden*26) = (277760, 832) intermediate tensor and a 26,624-param linear
    layer.  This caused ~2.3 GB peak activation memory during backward.

    Fix: replace flatten+Linear with AdaptiveAvgPool1d(pool_size) + Linear(hidden*pool_size).
    With pool_size=4 this compresses time to 4 anchor points, reducing the linear input
    from 832 → 128 and peak activation memory from ~2.3 GB to ~600 MB.
    """
    # Number of temporal anchor points after pooling. Tune: 4 gives
    # roughly T/7 resolution (early/late morning, afternoon, night).
    POOL_SIZE = 4

    def __init__(self, in_features, hidden_dim, out_features, total_time=26):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, hidden_dim)

        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, hidden_dim)

        # Compress the time dimension to POOL_SIZE anchor points instead of
        # flattening all total_time steps.  This is the key memory fix.
        self.pool = nn.AdaptiveAvgPool1d(self.POOL_SIZE)

        compress_dim = hidden_dim * self.POOL_SIZE   # e.g. 32 * 4 = 128
        self.fc   = nn.Linear(compress_dim, out_features)
        self.norm3 = nn.LayerNorm(out_features)

    def forward(self, x):
        # x: (B*H*W, in_features, total_time)
        x = F.relu(self.norm1(self.conv1(x)))        # (B*H*W, hidden, T)
        x = F.relu(self.norm2(self.conv2(x)))        # (B*H*W, hidden, T)
        x = self.pool(x)                             # (B*H*W, hidden, POOL_SIZE)
        x = x.reshape(x.size(0), -1)                # (B*H*W, hidden*POOL_SIZE)
        return self.norm3(self.fc(x))                # (B*H*W, out_features)


class WNOBlock(nn.Module):
    """
    Wavelet Neural Operator Block.

    Two-stage spectral mixing from V2 is preserved:
      1. spectral_spatial  – 3×3 depthwise per sub-band (local spatial context)
      2. spectral_pointwise – 1×1 across all sub-bands (cross-frequency mixing)
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dwt  = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)

        self.spectral_spatial    = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        self.spectral_pointwise  = nn.Conv2d(width * 4, width * 4, kernel_size=1)

        self.spatial_mixer = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        self.pointwise     = nn.Conv2d(width, width, 1)
        self.norm          = nn.GroupNorm(4, width)

    def forward(self, x):
        x_norm = self.norm(x)

        x_w    = self.dwt(x_norm)
        x_w    = self.spectral_spatial(x_w)
        x_w    = self.spectral_pointwise(x_w)
        out_w  = self.idwt(x_w)

        out_w  = self.spatial_mixer(out_w)
        out    = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)


class FNO2D(nn.Module):
    """
    WNO model kept under the FNO2D class name for import compatibility.

    Input tensor x: (B, H, W, C) where the channel layout is:
        [0 : time_input]                          → PM2.5 history (10 ch)
        [time_input : time_input+T*num_temporal]  → temporal weather features (260 ch)
        [time_input+T*num_temporal : -1]          → static emissions aggregated (21 ch)
        [-1]                                      → topography proxy (1 ch)
    """
    def __init__(
        self,
        in_channels,
        time_out=16,
        width=64,
        modes=None,          # kept for API compat, not used by WNO
        time_input=10,
        total_time=26,
        num_temporal_features=10,
    ):
        super().__init__()
        self.width                = width
        self.time_out             = time_out
        self.time_input           = time_input
        self.total_time           = total_time
        self.num_temporal_features = num_temporal_features

        self.temporal_encoder = TemporalEncoder(
            in_features=num_temporal_features,
            hidden_dim=16,    # Fix 1b: halve hidden_dim 32→16, reduces TE backward footprint 4×
            out_features=32,
            total_time=total_time,
        )

        # After TemporalEncoder the temporal block shrinks from
        # (total_time * num_temporal_features) channels → 32 channels.
        original_temporal_channels = total_time * num_temporal_features
        static_topo_channels       = in_channels - time_input - original_temporal_channels
        new_in_channels            = time_input + 32 + static_topo_channels  # e.g. 10+32+22 = 64

        self.input_encoder = nn.Sequential(
            nn.Conv2d(new_in_channels + 2, width, kernel_size=1),  # +2 for grid coords
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.30),   # Fix 4c: raise dropout 0.20→0.30 for better regularisation
        )

        self.block0 = WNOBlock(width)
        self.block1 = WNOBlock(width)
        self.block2 = WNOBlock(width)
        self.block3 = WNOBlock(width)

        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, time_out, kernel_size=1)

    # ------------------------------------------------------------------
    def get_grid(self, b, nx, ny, device):
        gx = torch.linspace(0, 1, nx, device=device).view(1, 1, nx, 1).expand(b, 1, nx, ny)
        gy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, ny).expand(b, 1, nx, ny)
        return torch.cat([gx, gy], dim=1)

    def forward(self, x):
        b, nx, ny, _ = x.shape

        last_pm25 = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)

        c0 = self.time_input
        c1 = c0 + self.total_time * self.num_temporal_features

        pm          = x[..., :c0]              # (B, H, W, 10)
        temporal    = x[..., c0:c1]            # (B, H, W, T*F)
        static_topo = x[..., c1:]              # (B, H, W, 22)

        # ---- TemporalEncoder ----
        # Reshape to (B*H*W, num_temporal_features, total_time) for Conv1d
        temporal = temporal.reshape(b, nx, ny, self.total_time, self.num_temporal_features)
        temporal_flat = temporal.permute(0, 1, 2, 4, 3).reshape(
            b * nx * ny, self.num_temporal_features, self.total_time
        )
        # Fix 1a: run TemporalEncoder in float32 even inside AMP; prevents
        # the massive fp16 backward graph that poisons WNO block gradients.
        with torch.autocast(device_type=temporal_flat.device.type, enabled=False):
            temporal_encoded = self.temporal_encoder(temporal_flat.float())  # (B*H*W, 32)
        temporal_encoded = temporal_encoded.reshape(b, nx, ny, 32)

        x_new = torch.cat([pm, temporal_encoded, static_topo], dim=-1)  # (B, H, W, 64)
        x_in  = x_new.permute(0, 3, 1, 2)                               # (B, 64, H, W)
        grid  = self.get_grid(b, nx, ny, x.device)
        x_in  = torch.cat([x_in, grid], dim=1)                          # (B, 66, H, W)

        x_feat = self.input_encoder(x_in)

        x_wno = self.block0(x_feat)
        x_wno = self.block1(x_wno)
        x_wno = self.block2(x_wno)
        x_wno = self.block3(x_wno)

        x_wno = F.gelu(self.fc1(x_wno))
        out   = self.fc2(x_wno)

        # Residual: network learns Δ from last known PM2.5 state
        out = out + last_pm25                                            # broadcasts over 16 steps

        return out.permute(0, 2, 3, 1)                                   # (B, H, W, time_out)