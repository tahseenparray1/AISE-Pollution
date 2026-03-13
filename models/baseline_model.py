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
        
        # Depthwise conv with large kernel to expand receptive field
        self.spatial_mixer = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        
        self.pointwise = nn.Conv2d(width, width, 1)
        # FIX: InstanceNorm prevents cross-channel spatial blurring of localized emission spikes
        self.norm = nn.InstanceNorm2d(width, affine=True)

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

class FNO2D(nn.Module):
    """
    Encoder-Decoder WNO architecture.
    
    Encoder: Processes historical context (PM2.5 history + historical weather + emissions + topo)
             to learn the current pollution state.
    Decoder: Conditions on future weather features to predict PM2.5 evolution.
    
    Input layout (channel dim): [encoder_inputs | decoder_inputs]
      - Encoder: PM hist (10) + hist weather 10h×10 (100) + emissions (7) + topo (1) = 118
      - Decoder: Future weather 16h×10 = 160
      - Total: 278 channels (same as flat architecture)
    """
    def __init__(self, enc_channels, dec_channels, time_out=16, width=64, modes=None, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        
        # --- ENCODER: learns current pollution state from historical data ---
        self.enc_input = nn.Sequential(
            nn.Conv2d(enc_channels + 2, width, kernel_size=1),  # +2 for spatial grid
            nn.InstanceNorm2d(width, affine=True),
            nn.GELU(),
            nn.Dropout(p=0.05)
        )
        self.enc_block0 = WNOBlock(self.width)
        self.enc_block1 = WNOBlock(self.width)
        
        # --- FUTURE CONDITIONING: projects future weather into latent space ---
        self.future_proj = nn.Sequential(
            nn.Conv2d(dec_channels, width, kernel_size=1),
            nn.GELU()
        )
        
        # --- DECODER: predicts PM2.5 evolution under future weather conditions ---
        # Explicit state injection: gives decoder the exact T=0 state to evolve
        self.dec_input = nn.Sequential(
            nn.Conv2d(width + 1, width, kernel_size=1),  # +1 for last_pm25
            nn.GELU()
        )
        self.dec_block0 = WNOBlock(self.width)
        self.dec_block1 = WNOBlock(self.width)
        
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
        
        # Extract last known PM2.5 state for residual connection (still channel 9)
        last_pm25 = x[..., self.time_input-1:self.time_input].permute(0, 3, 1, 2)
        
        # Split input into encoder and decoder channels
        x_enc = x[..., :self.enc_channels].permute(0, 3, 1, 2)   # (B, enc_ch, H, W)
        x_dec = x[..., self.enc_channels:].permute(0, 3, 1, 2)   # (B, dec_ch, H, W)
        
        grid = self.get_grid(b, nx, ny, x.device)
        
        # --- ENCODE: understand current pollution state ---
        h = self.enc_input(torch.cat([x_enc, grid], dim=1))
        h = self.enc_block0(h)
        h = self.enc_block1(h)
        
        # --- CONDITION: inject future weather information ---
        future_cond = self.future_proj(x_dec)
        h = h + future_cond  # additive conditioning
        
        # --- DECODE: explicit physical injection of starting state ---
        # The decoder now has direct access to the exact T=0 pollution field it needs to evolve
        h = self.dec_input(torch.cat([h, last_pm25], dim=1))
        
        h = self.dec_block0(h)
        h = self.dec_block1(h)
        
        # --- OUTPUT ---
        h = F.gelu(self.fc1(h))
        out = self.fc2(h)
        
        # Residual: network learns DELTA from last known PM2.5 state
        out = out + last_pm25
        
        return out.permute(0, 2, 3, 1)