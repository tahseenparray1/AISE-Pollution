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

class WNOBlock(nn.Module):
    """ Wavelet Neural Operator Block with Extended Receptive Field """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dwt = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)
        
        self.spectral_mixer = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        self.spatial_mixer = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        x_norm = self.norm(x)
        x_w = self.dwt(x_norm)
        x_w = self.spectral_mixer(x_w)
        out_w = self.idwt(x_w)
        out_w = self.spatial_mixer(out_w)
        out = out_w + self.pointwise(x_norm)
        return x + F.gelu(out)

class FiLMGenerator(nn.Module):
    """ 
    Generates scale (gamma) and shift (beta) parameters from the Topography map.
    This conditions the global wind advection based on the Indian geography.
    """
    def __init__(self, target_channels):
        super().__init__()
        # Downsample topography from 140x124 to 70x62 to match the W-UNet bottleneck
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, target_channels * 2, kernel_size=3, padding=1)
        )

    def forward(self, topo):
        params = self.net(topo) # Output shape: (B, 2 * width, 70, 62)
        gamma, beta = torch.chunk(params, 2, dim=1) # Split into scale and shift
        return gamma, beta

class WaveletUNet(nn.Module):
    """
    Topography-Conditioned Wavelet U-Net (W-UNet)
    """
    def __init__(self, in_channels, time_out=16, width=128, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input  
        
        # NOTE: (in_channels - 1) because we pull the topography channel out for FiLM!
        self.inc = nn.Sequential(
            nn.Conv2d((in_channels - 1) + 2, width // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, width // 2),
            nn.GELU()
        )
        
        self.down = nn.Sequential(
            nn.Conv2d(width // 2, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU()
        )
        
        self.wno_blocks = nn.Sequential(
            WNOBlock(width),
            WNOBlock(width),
            WNOBlock(width)
        )
        
        # Topography Modulator
        self.film = FiLMGenerator(target_channels=width)
        
        self.up = nn.ConvTranspose2d(width, width // 2, kernel_size=2, stride=2)
        
        self.dec = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=3, padding=1), 
            nn.GroupNorm(4, width // 2),
            nn.GELU(),
            nn.Dropout(p=0.1) 
        )
        
        self.fc_out = nn.Conv2d(width // 2, time_out, kernel_size=1)

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1) 

    def forward(self, x):
        b, nx, ny, _ = x.shape
        last_pm25 = x[..., self.time_input-1:self.time_input].permute(0, 3, 1, 2) 
        
        x_in = x.permute(0, 3, 1, 2)
        
        # 1. Split Topography from Main Features
        topo = x_in[:, -1:, :, :]       # The 1 topography channel
        x_main = x_in[:, :-1, :, :]     # The 277 temporal/emission channels
        
        # 2. Add Coordinate Grid to main features
        grid = self.get_grid(b, nx, ny, x.device) 
        x_main = torch.cat([x_main, grid], dim=1) 
        
        # 3. Encoder -> Downsample
        x_skip = self.inc(x_main)     
        x_down = self.down(x_skip)    
        
        # 4. Generate Topography FiLM parameters
        gamma, beta = self.film(topo)  # Shapes: (B, 128, 70, 62)
        
        # 5. Bottleneck processing
        x_wno = self.wno_blocks(x_down) 
        
        # --- APPLY TOPOGRAPHY CONDITIONING ---
        # Scales and shifts the WNO latents based on geographic boundaries
        x_wno = x_wno * (1 + gamma) + beta
        
        # 6. Upsample -> Decode
        x_up = self.up(x_wno)          
        x_concat = torch.cat([x_up, x_skip], dim=1) 
        
        x_dec = self.dec(x_concat)
        out = self.fc_out(x_dec)
        
        # 7. Delta Residual Connection
        out = out + last_pm25
        
        return out.permute(0, 2, 3, 1)