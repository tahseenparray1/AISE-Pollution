import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNOBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.pointwise = nn.Conv2d(width, width, 1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))

class FNO2D(nn.Module):
    def __init__(self, in_channels, time_out, width=64, modes=12):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # ==========================================
        # 1. TEMPORAL 3D ENCODERS 
        # ==========================================
        # PM2.5 History: (1 Feature over 10 hours)
        # We use a 3x1x1 kernel to calculate temporal gradients, then collapse to 1 time step
        self.pm_encoder = nn.Sequential(
            nn.Conv3d(1, width // 4, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Conv3d(width // 4, width // 2, kernel_size=(10, 1, 1))
        )
        
        # Weather/Emissions: (15 physical features + 2 grid coordinates = 17)
        # We evaluate the sequence of 26 hours, then collapse it cleanly
        self.weather_encoder = nn.Sequential(
            nn.Conv3d(17, width // 2, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GELU(),
            nn.Conv3d(width // 2, width // 2, kernel_size=(26, 1, 1))
        )
        
        # Spatial FNO Blocks
        self.block0 = FNOBlock(self.width, self.modes)
        self.block1 = FNOBlock(self.width, self.modes)
        self.block2 = FNOBlock(self.width, self.modes)
        self.block3 = FNOBlock(self.width, self.modes)
        
        self.dropout = nn.Dropout2d(p=0.1) 
        
        # Decoders
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, time_out)

    def get_grid(self, b, nx, ny, device):
        """Generates static grid, expanded to match the 26-hour temporal dimension"""
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, 1, nx, 1).repeat(b, 1, 26, 1, ny)
        gridy = gridy.view(1, 1, 1, 1, ny).repeat(b, 1, 26, nx, 1)
        return torch.cat((gridx, gridy), dim=1) # (b, 2, 26, nx, ny)

    def forward(self, x):
        b, nx, ny, _ = x.shape
        
        # ==========================================
        # 2. UNPACK & RESHAPE 3D INPUTS
        # ==========================================
        # PM2.5: (b, 1, 10, nx, ny)
        pm25 = x[..., :10].permute(0, 3, 1, 2).unsqueeze(1)       
        
        # Weather: Unpack the 390 channels back into physical reality (15 features, 26 hours)
        weather_flat = x[..., 10:] 
        weather_3d = weather_flat.view(b, nx, ny, 26, 15).permute(0, 4, 3, 1, 2) # (b, 15, 26, nx, ny)
        
        # Save absolute last known state of PM2.5 for the residual (Hour 10)
        last_pm25 = x[..., 9:10] # (b, nx, ny, 1)
        
        # Inject Geographic Memory across all 26 hours
        grid = self.get_grid(b, nx, ny, x.device)
        weather_with_grid = torch.cat([weather_3d, grid], dim=1) # (b, 17, 26, nx, ny)
        
        # ==========================================
        # 3. 3D FEATURE EXTRACTION
        # ==========================================
        # The Conv3d layers process the temporal gradients and squeeze the time dimension out
        pm_feat = self.pm_encoder(pm25).squeeze(2)                        # (b, width/2, nx, ny)
        weather_feat = self.weather_encoder(weather_with_grid).squeeze(2) # (b, width/2, nx, ny)
        
        x_lifted = torch.cat([pm_feat, weather_feat], dim=1)              # (b, width, nx, ny)
        
        # ==========================================
        # 4. OPTIMIZED FFT PADDING (144x128)
        # ==========================================
        # nx=140 + (2+2) = 144
        # ny=124 + (2+2) = 128
        # These dimensions are highly optimal for spectral transforms
        pad_x, pad_y = 2, 2
        x_padded = F.pad(x_lifted, (pad_y, pad_y, pad_x, pad_x)) 
        
        # ==========================================
        # 5. SPATIAL FOURIER OPERATOR BLOCKS
        # ==========================================
        x_fno = self.block0(x_padded)
        x_fno = self.block1(x_fno)       
        x_fno = self.block2(x_fno)
        x_fno = self.block3(x_fno)       
        
        # Unpad symmetrically back to 140x124
        x_fno = x_fno[..., pad_x:-pad_x, pad_y:-pad_y]
        x_fno = self.dropout(x_fno)
        
        # ==========================================
        # 6. PROJECTION & RESIDUAL
        # ==========================================
        x_fno = x_fno.permute(0, 2, 3, 1) # (b, nx, ny, width)
        x_fno = F.gelu(self.fc1(x_fno))
        delta = self.fc2(x_fno)           # (b, nx, ny, 16)
        
        # Add residual to predict the physical target
        out = last_pm25 + delta
        return out