import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        
        # Initialize complex parameters safely for any PyTorch version
        self.weights1 = nn.Parameter(self.scale * torch.complex(
            torch.randn(in_channels, out_channels, self.modes1, self.modes2),
            torch.randn(in_channels, out_channels, self.modes1, self.modes2)
        ))
        self.weights2 = nn.Parameter(self.scale * torch.complex(
            torch.randn(in_channels, out_channels, self.modes1, self.modes2),
            torch.randn(in_channels, out_channels, self.modes1, self.modes2)
        ))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Ensure modes do not exceed the actual input size
        m1 = min(self.modes1, x.size(-2) // 2)
        m2 = min(self.modes2, x.size(-1) // 2 + 1)
        
        out_ft[:, :, :m1, :m2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock(nn.Module):
    """ True Fourier Neural Operator Block """
    def __init__(self, width, modes1=8, modes2=8):
        super().__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        x_norm = self.norm(x)
        x1 = self.conv(x_norm)
        x2 = self.w(x_norm)
        return x + F.gelu(x1 + x2)

class FiLM(nn.Module):
    """ Feature-wise Linear Modulation for conditional encoding """
    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cond_channels, feature_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels * 2, feature_channels * 2, kernel_size=3, padding=1)
        )
    def forward(self, x, cond):
        cond_out = self.net(cond)
        gamma, beta = torch.chunk(cond_out, 2, dim=1)
        return x * (1 + gamma) + beta

class FNO2D(nn.Module):
    def __init__(self, in_channels, time_out=16, width=64, modes=8):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.modes = modes if modes is not None else 8
        
        # 1. Spatial-Temporal Disentanglement: Initial temporal/channel mixing instead of 3D Conv
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1), # Explicit Temporal Channel Mix
            nn.GELU(),
            nn.Dropout2d(p=0.1) 
        )
        
        # 2. Conditional Encoding (Grid context via FiLM)
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(2, width // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1)
        )
        self.film1 = FiLM(width // 2, width)
        self.film2 = FiLM(width // 2, width)
        
        # 3. U-Net Backbone with True FNO
        # Encoder Level 1
        self.fno1 = FNOBlock(width, self.modes, self.modes)
        
        # Downsample to Level 2
        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, width * 2),
            nn.GELU()
        )
        
        # Bottleneck FNO layers
        self.fno2 = FNOBlock(width * 2, self.modes, self.modes)
        self.fno3 = FNOBlock(width * 2, self.modes, self.modes)
        
        # Upsample back to Level 1
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width * 2, width, kernel_size=2, stride=2),
            nn.GroupNorm(4, width),
            nn.GELU()
        )
        
        # Skip connection mixer
        self.skip_mixer = nn.Conv2d(width * 2, width, kernel_size=1)
        
        # Decoder Level 1
        self.fno4 = FNOBlock(width, self.modes, self.modes)
        
        # Output Head
        self.fc1 = nn.Conv2d(width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, self.time_out, kernel_size=1) 

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        b, nx, ny, _ = x.shape
        
        # Extract inputs and grid
        x_in = x.permute(0, 3, 1, 2)
        grid = self.get_grid(b, nx, ny, x.device) 
        
        # Grid conditioning using FiLM
        cond = self.cond_encoder(grid)
        
        # Initial temporal/feature encoding
        x_feat = self.input_encoder(x_in)
        
        # Add conditioning early
        x_feat = self.film1(x_feat, cond)
        
        # U-Net Encoder
        x1 = self.fno1(x_feat)       # -> [B, width, H, W]
        
        # Downsample
        x2 = self.down(x1)           # -> [B, width*2, H/2, W/2]
        
        # Bottleneck
        x2 = self.fno2(x2)
        x2 = self.fno3(x2)
        
        # Upsample
        x3 = self.up(x2)             # -> [B, width, H, W]
        
        # Skip connection from the encoder
        x_cat = torch.cat([x3, x1], dim=1) # -> [B, width*2, H, W]
        x_mixed = self.skip_mixer(x_cat)
        
        # Add conditioning late
        x_mixed = self.film2(x_mixed, cond)
        
        # Final spatial FNO
        x_out = self.fno4(x_mixed)
        
        # Decode to temporal predictions
        x_out = F.gelu(self.fc1(x_out))
        out = self.fc2(x_out) 
        
        return out.permute(0, 2, 3, 1)