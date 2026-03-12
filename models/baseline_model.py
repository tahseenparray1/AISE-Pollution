import torch
import torch.nn as nn
import torch.nn.functional as F


class FastTimeChannelBlock(nn.Module):
    """
    Replaces Conv3D by treating temporal features as channels.
    Input shape: (B, width, H, W) — purely 2D operations.
    
    Architecture:
    - Conv2d k=3: local spatial mixing
    - Depthwise Conv2d k=5: large-kernel wind advection
    - Conv2d k=1: cross-channel (cross-time) mixing
    - Residual connection
    """
    def __init__(self, width):
        super().__init__()
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            # Large receptive field for wind advection without 3D overhead
            nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width),
            nn.GroupNorm(4, width)
        )
        # Cross-channel mixing acts as temporal mixer
        self.channel_mixer = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x):
        out = self.spatial_mixer(x)
        out = self.channel_mixer(out)
        return x + F.gelu(out)


class FNO2D(nn.Module):
    """
    Time-as-Channel 2D Spatial Operator.
    
    Instead of 3D convolutions, flattens the T=10 input hours into channels:
        (B, C, T=10, H, W) → (B, C*10, H, W)
    Then applies fast, highly-optimized 2D convolutions.
    
    This avoids:
    - The O(T*H*W*C*k_t*k_h*k_w) cost of 3D convolutions
    - Haar wavelet checkerboarding artifacts
    - Artificial gradient cliffs from zero-masked future hours
    
    Input:  (B, C_in, T, H, W) — 3D spatiotemporal field
    Output: (B, H, W, time_out) — predicted PM2.5 for future hours
    """
    def __init__(self, in_channels, time_out=16, width=128, modes=None,
                 time_input=10, time_steps=26):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input
        
        # Flatten time into channels: C_in * T_in + 2 grid coords
        flat_channels = in_channels * time_input
        
        self.input_encoder = nn.Sequential(
            nn.Conv2d(flat_channels + 2, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.05)
        )
        
        # 4 spatial mixing blocks
        self.block0 = FastTimeChannelBlock(self.width)
        self.block1 = FastTimeChannelBlock(self.width)
        self.block2 = FastTimeChannelBlock(self.width)
        self.block3 = FastTimeChannelBlock(self.width)
        
        # Output decoder
        self.fc1 = nn.Conv2d(self.width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, self.time_out, kernel_size=1)

    def get_grid(self, b, nx, ny, device):
        """2D positional grid: (gridx, gridy) → (B, 2, H, W)"""
        gridx = torch.linspace(0, 1, nx, device=device).view(1, 1, nx, 1).expand(b, 1, nx, ny)
        gridy = torch.linspace(0, 1, ny, device=device).view(1, 1, 1, ny).expand(b, 1, nx, ny)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        # x: (B, C_in, T, H, W)
        b, c, t, nx, ny = x.shape
        
        # Only use first T_in hours — avoids the artificial gradient cliff
        x_in = x[:, :, :self.time_input, :, :]  # (B, C, 10, H, W)
        
        # Extract last known PM2.5 for residual (channel 0, last input hour)
        last_pm25 = x_in[:, 0, -1, :, :]  # (B, H, W)
        
        # Flatten Time into Channels: (B, C*T_in, H, W)
        x_flat = x_in.reshape(b, c * self.time_input, nx, ny)
        
        # Append spatial grid
        grid = self.get_grid(b, nx, ny, x.device)  # (B, 2, H, W)
        x_flat = torch.cat([x_flat, grid], dim=1)
        
        # Encode
        x_feat = self.input_encoder(x_flat)
        
        # Spatial mixing (pure 2D — blazing fast)
        x_feat = self.block0(x_feat)
        x_feat = self.block1(x_feat)
        x_feat = self.block2(x_feat)
        x_feat = self.block3(x_feat)
        
        # Decode to 16 future hours
        x_feat = F.gelu(self.fc1(x_feat))   # (B, 128, H, W)
        out = self.fc2(x_feat)               # (B, 16, H, W)
        
        # Residual: anchor to last known PM2.5
        out = out + last_pm25.unsqueeze(1)   # broadcast (B, 1, H, W)
        
        # Return (B, H, W, 16) for loss compatibility
        return out.permute(0, 2, 3, 1)