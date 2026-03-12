import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWavelet2D(nn.Module):
    """ Native PyTorch 2D Haar Discrete Wavelet Transform """
    def __init__(self, in_channels):
        super().__init__()
        h0 = [1/2, 1/2]
        h1 = [-1/2, 1/2]
        
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
    """ Native PyTorch 2D Inverse Haar Wavelet Transform """
    def __init__(self, in_channels):
        super().__init__()
        h0 = [1.0, 1.0]
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

class WNOBlock3D(nn.Module):
    """
    3D Spatiotemporal Wavelet Neural Operator Block.
    
    Uses 2D Haar wavelets for spatial multi-resolution (applied per-timestep),
    then Conv3d for temporal-spatial coupling.
    
    Input/Output shape: (B, width, T, H, W)
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        
        # Spatial wavelets (applied per-timestep by folding T into batch)
        self.dwt = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)
        
        # 3D spectral mixer: mixes wavelet sub-bands AND time simultaneously
        self.spectral_mixer = nn.Conv3d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        
        # 3D spatial-temporal mixer: kernel (3,5,5) sweeps across time + space
        # This is where the "arrow of time" enters — the model learns how
        # pollution advects from hour t to hour t+1 across the spatial grid
        self.spatial_mixer = nn.Conv3d(width, width, kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=width)
        
        self.pointwise = nn.Conv3d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x_norm = self.norm(x)
        
        # 1. Fold T into batch for 2D Haar: (B*T, C, H, W)
        x_flat = x_norm.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # 2. Spatial wavelet decomposition (per-timestep)
        x_w = self.dwt(x_flat)  # (B*T, C*4, H/2, W/2)
        
        # 3. Unfold back to 3D: (B, C*4, T, H/2, W/2)
        _, C4, Hh, Wh = x_w.shape
        x_w = x_w.reshape(B, T, C4, Hh, Wh).permute(0, 2, 1, 3, 4)
        
        # 4. 3D spectral mixing (mixes sub-bands + time)
        x_w = self.spectral_mixer(x_w)
        
        # 5. Fold + Inverse Haar + unfold
        x_w = x_w.permute(0, 2, 1, 3, 4).reshape(B * T, C4, Hh, Wh)
        out_w = self.idwt(x_w)  # (B*T, C, H, W)
        out_w = out_w.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # 6. 3D spatial-temporal mixing: the temporal advection kernel
        out_w = self.spatial_mixer(out_w)
        
        # 7. Pointwise + residual
        out = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)


class FNO2D(nn.Module):
    """
    3D Spatiotemporal Wavelet Neural Operator.
    
    Name kept as FNO2D for backward-compatible imports.
    
    Input:  (B, C_in, T, H, W) — 3D spatiotemporal field
    Output: (B, H, W, time_out) — predicted PM2.5 for future hours
    """
    def __init__(self, in_channels, time_out=16, width=64, modes=None, 
                 time_input=10, time_steps=26):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input
        self.time_steps = time_steps
        
        # 3D Input encoder: C_in + 3 grid coords → width
        self.input_encoder = nn.Sequential(
            nn.Conv3d(in_channels + 3, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.05)
        )
        
        # 4 WNO3D blocks
        self.block0 = WNOBlock3D(self.width)
        self.block1 = WNOBlock3D(self.width)
        self.block2 = WNOBlock3D(self.width)
        self.block3 = WNOBlock3D(self.width)
        
        # Output decoder: width → 1 channel (PM2.5)
        self.fc1 = nn.Conv3d(self.width, 128, kernel_size=1)
        self.fc2 = nn.Conv3d(128, 1, kernel_size=1)

    def get_grid(self, b, t, nx, ny, device):
        """3D positional grid: (gridx, gridy, gridt) → (B, 3, T, H, W)"""
        gridx = torch.linspace(0, 1, nx, device=device).view(1, 1, 1, nx, 1).expand(b, 1, t, nx, ny)
        gridy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, 1, ny).expand(b, 1, t, nx, ny)
        gridt = torch.linspace(0, 1, t, device=device).view(1, 1, t, 1, 1).expand(b, 1, t, nx, ny)
        return torch.cat((gridx, gridy, gridt), dim=1)  # (B, 3, T, H, W)

    def forward(self, x):
        # x: (B, C_in, T, H, W)
        b, c, t, nx, ny = x.shape
        
        # Extract last known PM2.5 for residual (channel 0 = PM2.5, hour 9 = last input)
        # PM2.5 is channel index 0 (first channel in the data loader)
        last_pm25 = x[:, 0, self.time_input - 1, :, :]  # (B, H, W)
        
        # Append 3D positional grid
        grid = self.get_grid(b, t, nx, ny, x.device)  # (B, 3, T, H, W)
        x_in = torch.cat([x, grid], dim=1)  # (B, C_in+3, T, H, W)
        
        # Encode
        x_feat = self.input_encoder(x_in)  # (B, width, T, H, W)
        
        # 3D WNO blocks
        x_wno = self.block0(x_feat)
        x_wno = self.block1(x_wno)
        x_wno = self.block2(x_wno)
        x_wno = self.block3(x_wno)
        
        # Decode to single-channel 26-hour prediction
        x_wno = F.gelu(self.fc1(x_wno))   # (B, 128, T, H, W)
        out = self.fc2(x_wno)              # (B, 1, T, H, W)
        out = out.squeeze(1)               # (B, T, H, W)
        
        # Slice future hours: hours 10-25 = the 16 prediction targets
        out = out[:, self.time_input:, :, :]  # (B, 16, H, W)
        
        # Residual connection: add last known PM2.5 to all future steps
        out = out + last_pm25.unsqueeze(1)  # broadcast (B, 1, H, W) → (B, 16, H, W)
        
        # Return in (B, H, W, T_out) format for compatibility with loss code
        return out.permute(0, 2, 3, 1)