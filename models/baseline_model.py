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
    """ Wavelet Neural Operator Block """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dwt = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)
        
        # Mixes the 4 wavelet sub-bands (LL, LH, HL, HH)
        self.spectral_mixer = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        
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
        
        # 4. Add spatial point-wise mixing and residual
        out = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)

class FNO2D(nn.Module):
    # Renamed internal structure to WNO, kept class name FNO2D to prevent breaking your train.py imports!
    def __init__(self, in_channels, time_out=16, width=64, modes=None):
        super().__init__()
        self.width = width
        self.time_out = time_out
        
        # Encode ablated input down to 'width'
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 2, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout2d(p=0.1) 
        )
        
        # Stack WNO blocks instead of FNO blocks
        self.block0 = WNOBlock(self.width)
        self.block1 = WNOBlock(self.width)
        self.block2 = WNOBlock(self.width)
        self.block3 = WNOBlock(self.width)
        
        self.fc1 = nn.Conv2d(self.width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, self.time_out, kernel_size=1) 

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1) 

    def forward(self, x):
        b, nx, ny, _ = x.shape
        last_pm25 = x[..., 9:10].permute(0, 3, 1, 2) 
        
        x_in = x.permute(0, 3, 1, 2)
        grid = self.get_grid(b, nx, ny, x.device) 
        x_in = torch.cat([x_in, grid], dim=1) 
        
        x_feat = self.input_encoder(x_in)
        
        # WNO Spatial Mixing (No padding needed, 140 and 124 divide evenly by 2!)
        x_wno = self.block0(x_feat)
        x_wno = self.block1(x_wno)
        x_wno = self.block2(x_wno)
        x_wno = self.block3(x_wno)
        
        # Decode to DELTA
        x_wno = F.gelu(self.fc1(x_wno))
        out = self.fc2(x_wno) 
        
        # RESIDUAL CONNECTION
        
        return out.permute(0, 2, 3, 1)