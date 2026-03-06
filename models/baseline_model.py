import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
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
        
        # +2 for the spatial (x, y) positional encoding
        self.fc0 = nn.Linear(in_channels + 2, self.width)
        
        self.block0 = FNOBlock(self.width, self.modes)
        self.block1 = FNOBlock(self.width, self.modes)
        self.block2 = FNOBlock(self.width, self.modes)
        self.block3 = FNOBlock(self.width, self.modes)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, time_out)

    def forward(self, x):
        # x shape: (batch, nx, ny, in_channels)
        b, nx, ny, _ = x.shape
        
        grid = self.get_grid(b, nx, ny, x.device)
        x = torch.cat((x, grid), dim=-1) # Append spatial coordinates
        
        x = self.fc0(x) # Project to hidden width
        x = x.permute(0, 3, 1, 2) # (batch, width, nx, ny) for 2D Convs
        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.permute(0, 2, 3, 1) # Back to (batch, nx, ny, width) for Linear layers
        x = F.gelu(self.fc1(x))
        x = self.fc2(x) # Project to time_out
        
        return x

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, nx, 1, 1).repeat(b, 1, ny, 1)
        gridy = gridy.view(1, 1, ny, 1).repeat(b, nx, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)