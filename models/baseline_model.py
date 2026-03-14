import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWavelet2D(nn.Module):
    """ Native PyTorch 2D Haar Discrete Wavelet Transform """
    def __init__(self, in_channels):
        super().__init__()
        # Haar wavelet filters: Low-Low, Low-High, High-Low, High-High
        h0 = [1/2, 1/2]
        h1 = [-1/2, 1/2]
        
        # Create 2D filters via outer product
        ll = torch.tensor([[h0[0]*h0[0], h0[0]*h0[1]], [h0[1]*h0[0], h0[1]*h0[1]]])
        lh = torch.tensor([[h1[0]*h0[0], h1[0]*h0[1]], [h1[1]*h0[0], h1[1]*h0[1]]])
        hl = torch.tensor([[h0[0]*h1[0], h0[0]*h1[1]], [h0[1]*h1[0], h0[1]*h1[1]]])
        hh = torch.tensor([[h1[0]*h1[0], h1[0]*h1[1]], [h1[1]*h1[0], h1[1]*h1[1]]])
        
        filters = torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('filters', filters)
        self.in_channels = in_channels
        
    def forward(self, x):
        # Applies DWT using stride 2 convolution
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
        # Reconstructs using transposed convolution
        return F.conv_transpose2d(x, self.filters, stride=2, groups=self.in_channels)

class WNOBlock(nn.Module):
    """ Wavelet Neural Operator Block with Extended Receptive Field """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dwt = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)
        
        # Mixes the 4 wavelet sub-bands (LL, LH, HL, HH)
        self.spectral_mixer = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        
        # FIX #4: Depthwise conv with large kernel to expand receptive field
        # Enables modeling long-range wind advection across the 140x124 spatial grid
        self.spatial_mixer = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        x_norm = self.norm(x)
        
        # 1. Decompose to Wavelet domain
        x_w = self.dwt(x_norm)
        
        # 2. Mix frequencies
        x_w = self.spectral_mixer(x_w)
        
        # 3. Reconstruct back to Spatial domain
        out_w = self.idwt(x_w)
        
        # 4. Apply large-kernel spatial mixing for extended receptive field
        out_w = self.spatial_mixer(out_w)
        
        # 5. Add pointwise mixing and residual
        out = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)

class WaveletUNet(nn.Module):
    """
    Topography-Conditioned Wavelet U-Net (W-UNet)
    Replaces the flat FNO2D to increase receptive field, stop overfitting, and capture spikes.
    """
    def __init__(self, in_channels, time_out=16, width=128, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input  
        
        # 1. Initial Encoder (Full Resolution: 140x124)
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels + 2, width // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, width // 2),
            nn.GELU()
        )
        
        # 2. Downsampling (Half Resolution: 70x62)
        # Stride 2 reduces spatial dimensions, preventing overfitting and boosting receptive field
        self.down = nn.Sequential(
            nn.Conv2d(width // 2, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU()
        )
        
        # 3. WNO Bottleneck (Global Physics Transport at 70x62)
        # Operates purely on the low-res spatial domain for global wind/advection
        self.wno_blocks = nn.Sequential(
            WNOBlock(width),
            WNOBlock(width),
            WNOBlock(width)
        )
        
        # 4. Upsampling (Back to Full Resolution: 140x124)
        self.up = nn.ConvTranspose2d(width, width // 2, kernel_size=2, stride=2)
        
        # 5. Decoder & Skip Connection Processing
        self.dec = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1), # width because of skip connection (width//2 + width//2)
            nn.GroupNorm(4, width // 2),
            nn.GELU(),
            nn.Dropout(p=0.1) # Increased dropout to fight overfitting
        )
        
        # 6. Final Projection to Delta Target
        self.fc_out = nn.Conv2d(width // 2, time_out, kernel_size=1)

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny, 1) if gridx.dim() == 4 else gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1) 

    def forward(self, x):
        b, nx, ny, _ = x.shape
        last_pm25 = x[..., self.time_input-1:self.time_input].permute(0, 3, 1, 2) 
        
        x_in = x.permute(0, 3, 1, 2)
        
        # Note: Added fixing grid issues in the original
        gridx = torch.linspace(0, 1, nx, device=x.device)
        gridy = torch.linspace(0, 1, ny, device=x.device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        grid = torch.cat((gridx, gridy), dim=1) 
        
        x_in = torch.cat([x_in, grid], dim=1) 
        
        # Encoder (Save for skip connection)
        x_skip = self.inc(x_in)       # Shape: (B, 64, 140, 124)
        
        # Downsample
        x_down = self.down(x_skip)    # Shape: (B, 128, 70, 62)
        
        # WNO Bottleneck
        x_wno = self.wno_blocks(x_down) # Shape: (B, 128, 70, 62)
        
        # Upsample
        x_up = self.up(x_wno)         # Shape: (B, 64, 140, 124)
        
        # Skip Connection (Concatenate along channel dim)
        x_concat = torch.cat([x_up, x_skip], dim=1) # Shape: (B, 128, 140, 124)
        
        # Decode
        x_dec = self.dec(x_concat)
        out = self.fc_out(x_dec)
        
        # Residual Connection (Predicting the Delta)
        out = out + last_pm25
        
        return out.permute(0, 2, 3, 1)