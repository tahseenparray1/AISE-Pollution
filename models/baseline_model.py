import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> GroupNorm -> SiLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FNO2D(nn.Module):
    """
    Time-as-Channel U-Net for PM2.5 forecasting.
    
    Architecture:
    - Encoder: 3 downsampling stages with skip connections
    - Bottleneck: global spatial context at 1/8 resolution
    - Decoder: 3 upsampling stages with skip concatenation + spatial alignment
    - Zero-initialized output head for persistence forecast start
    
    Input:  (B, C_in, T, H, W) — sliced to T=time_input internally
    Output: (B, H, W, time_out) — predicted PM2.5
    """
    def __init__(self, in_channels, time_out=16, width=96, modes=None,
                 time_input=10, time_steps=26):
        super().__init__()
        self.time_input = time_input
        self.time_out = time_out
        
        flat_channels = in_channels * time_input
        base_width = width
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(flat_channels, base_width)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_width, base_width * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_width * 2, base_width * 4))
        
        # Bottleneck (Global Spatial Context)
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_width * 4, base_width * 8))
        
        # Decoder (Upsampling + Skip Connections)
        self.up1 = nn.ConvTranspose2d(base_width * 8, base_width * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_width * 8, base_width * 4)
        
        self.up2 = nn.ConvTranspose2d(base_width * 4, base_width * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_width * 4, base_width * 2)
        
        self.up3 = nn.ConvTranspose2d(base_width * 2, base_width, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_width * 2, base_width)
        
        # Final projection to time_out future hours
        self.outc = nn.Conv2d(base_width, time_out, kernel_size=1)
        
        # Zero-initialize output head: model starts at persistence forecast
        # (out = 0 + last_pm25 = perfect persistence baseline at epoch 0)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        b, c, t, nx, ny = x.shape
        
        x_in = x[:, :, :self.time_input, :, :]
        last_pm25 = x_in[:, 0, -1, :, :]  # (B, H, W)
        
        x_flat = x_in.reshape(b, c * self.time_input, nx, ny)
        
        # Encoder with skip connections
        x1 = self.inc(x_flat)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with spatial alignment for odd dimensions (140x124)
        x_up1 = self.up1(x4)
        diffY = x3.size(2) - x_up1.size(2)
        diffX = x3.size(3) - x_up1.size(3)
        x_up1 = F.pad(x_up1, [diffX // 2, diffX - diffX // 2,
                               diffY // 2, diffY - diffY // 2])
        x_up1 = torch.cat([x3, x_up1], dim=1)
        x_up1 = self.conv_up1(x_up1)
        
        x_up2 = self.up2(x_up1)
        diffY = x2.size(2) - x_up2.size(2)
        diffX = x2.size(3) - x_up2.size(3)
        x_up2 = F.pad(x_up2, [diffX // 2, diffX - diffX // 2,
                               diffY // 2, diffY - diffY // 2])
        x_up2 = torch.cat([x2, x_up2], dim=1)
        x_up2 = self.conv_up2(x_up2)
        
        x_up3 = self.up3(x_up2)
        diffY = x1.size(2) - x_up3.size(2)
        diffX = x1.size(3) - x_up3.size(3)
        x_up3 = F.pad(x_up3, [diffX // 2, diffX - diffX // 2,
                               diffY // 2, diffY - diffY // 2])
        x_up3 = torch.cat([x1, x_up3], dim=1)
        x_up3 = self.conv_up3(x_up3)
        
        out = self.outc(x_up3)  # (B, 16, H, W) — starts at 0 due to zero-init
        out = out + last_pm25.unsqueeze(1)  # persistence baseline
        
        return out.permute(0, 2, 3, 1)  # (B, H, W, 16)