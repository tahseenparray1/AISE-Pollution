import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWavelet2D(nn.Module):
    """ Native PyTorch 2D Haar Discrete Wavelet Transform """
    def __init__(self, in_channels):
        super().__init__()
        # Haar wavelet filters: Low-Low, Low-High, High-Low, High-High
        h0 = [1 / 1.41421356, 1 / 1.41421356]
        h1 = [-1 / 1.41421356, 1 / 1.41421356]
        
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
        h0 = [1 / 1.41421356, 1 / 1.41421356]
        h1 = [-1 / 1.41421356, 1 / 1.41421356]
        
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
        self.spatial_mixer = nn.Sequential(
            # Spatial information (Depthwise)
            nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width),
            # Cross-channel information (Pointwise)
            nn.Conv2d(width, width, kernel_size=1)
        )
        
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        x_norm = self.norm(x)
        
        # 1. Decompose to Wavelet domain (Channels are interleaved: c1_ll, c1_lh, c2_ll...)
        x_w = self.dwt(x_norm)
        
        # 2. Reshape to group by sub-band (LLs together, LHs together, etc.)
        b, c_times_4, h, w = x_w.shape
        c = c_times_4 // 4
        # Reshape: (B, C, 4, H, W) -> Permute: (B, 4, C, H, W) -> Flatten: (B, 4*C, H, W)
        x_w = x_w.view(b, c, 4, h, w).permute(0, 2, 1, h, w).reshape(b, c_times_4, h, w)
        
        # 3. NOW mix frequencies (Groups=4 works perfectly here!)
        x_w = self.spectral_mixer(x_w)
        
        # 4. Reverse the permutation for IDWT
        x_w = x_w.view(b, 4, c, h, w).permute(0, 2, 1, h, w).reshape(b, c_times_4, h, w)
        
        # 5. Reconstruct back to Spatial domain
        out_w = self.idwt(x_w)
        
        # 4. Apply large-kernel spatial mixing for extended receptive field
        out_w = self.spatial_mixer(out_w)
        
        # 5. Add pointwise mixing and residual
        out = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)

class FNO2D(nn.Module):
    # Renamed internal structure to WNO, kept class name FNO2D to prevent breaking your train.py imports!
    def __init__(self, in_channels, time_out=16, width=64, modes=None, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input  # Number of PM2.5 history hours (for residual connection)
        
        # Encode ablated input down to 'width'
        # FIX #3: Use standard Dropout instead of Dropout2d to avoid
        # dropping entire temporal channels (e.g. "Hour 5 Wind Speed")
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 2, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout2d(p=0.01)
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
        # Extract last known PM2.5 state for residual connection
        last_pm25 = x[..., self.time_input-1:self.time_input].permute(0, 3, 1, 2) 
        
        x_in = x.permute(0, 3, 1, 2)
        grid = self.get_grid(b, nx, ny, x.device) 
        x_in = torch.cat([x_in, grid], dim=1) 
        
        x_feat = self.input_encoder(x_in)
        
        # WNO Spatial Mixing (No padding needed, 140 and 124 divide evenly by 2!)
        x_wno = self.block0(x_feat)
        x_wno = self.block1(x_wno)
        x_wno = self.block2(x_wno)
        x_wno = self.block3(x_wno)
        
        # Decode to DELTA (network learns the change from current state)
        x_wno = F.gelu(self.fc1(x_wno))
        out = self.fc2(x_wno) 
        
        # FIX #1: RESIDUAL CONNECTION
        # 'out' shape is (B, 16, H, W). 'last_pm25' shape is (B, 1, H, W).
        # Broadcasting adds the last known PM2.5 state to all 16 future steps.
        # This lets the network learn DELTA (change) instead of absolute values.
        out = out + last_pm25
        
        return out.permute(0, 2, 3, 1)