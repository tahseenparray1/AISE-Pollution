import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class SpectralConv2d_FNO(nn.Module):
    """
    2D Fourier spectral convolution layer.

    1. Computes FFT of the input over spatial dimensions (x, y)

    2. Multiplies low-frequency Fourier modes with learned complex-valued spectral kernels

    3. Sets higher-frequency modes to zero

    4. Applies inverse FFT to return to physical space

    This implements a global convolution operator parameterized in Fourier space.

    Parameters
    ----------
    width : int
        Number of input and output channels.

    modes1 : int
        Number of Fourier modes retained along x-direction.

    modes2 : int
        Number of Fourier modes retained along y-direction.

    Forward Input
    -------------
    x : torch.Tensor
        Shape (batch, channels, nx, ny, features)

    Returns
    -------
    out : torch.Tensor
        Shape (batch, channels, nx, ny, features)
    """

    def __init__(self, width, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / width

        self.weights = nn.Parameter(
            self.scale * torch.randn(
                width, width, modes1, modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, x, w):

        """
        Performs complex multiplication in Fourier space.

        Parameters
        ----------
        x : torch.Tensor
            Input Fourier coefficients of shape
            (batch, in_channels, modes1, modes2, features)

        w : torch.Tensor
            Complex spectral weights of shape
            (in_channels, out_channels, modes1, modes2)

        Returns
        -------
        out : torch.Tensor
            Output Fourier coefficients of shape
            (batch, out_channels, modes1, modes2, features)
        """

        return torch.einsum("bcxyf,cdxy->bdxyf", x, w)

    def forward(self, x):

        """
        Forward pass of the spectral convolution layer.

        1. Applies FFT over spatial dimensions

        2. Allocates an empty Fourier tensor for the output

        3. Applies learned spectral kernel multiplication on low-frequency modes

        4. Applies inverse FFT to obtain physical-space output

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, nx, ny, features)

        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch, channels, nx, ny, features)
        """
  
        b, c, nx, ny, f = x.shape

        x_ft = torch.fft.rfft2(x, dim=(2, 3))

        out_ft = torch.zeros(
            b, c, nx, ny // 2 + 1, f,
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2, :],
                self.weights[:, :, :self.modes1, :self.modes2]
            )

        x = torch.fft.irfft2(out_ft, s=(nx, ny), dim=(2, 3))
        return x



class FNOBlock(nn.Module):
    """
    Single Fourier Neural Operator block.

    1. Applies spectral convolution (global operator in Fourier space)

    2. Applies pointwise convolution in physical space

    3. Adds both results (skip connection)

    4. Applies GELU nonlinearity

    Parameters
    ----------
    width : int
        Number of hidden channels.

    features : int
        Number of feature dimensions per spatial location.

    modes : int
        Number of Fourier modes retained per spatial direction.

    Forward Input
    -------------
    x : torch.Tensor
        Shape (batch, width, nx, ny, features)

    Returns
    -------
    out : torch.Tensor
        Shape (batch, width, nx, ny, features)
    """

    def __init__(self, width,features,modes):
        super().__init__()
        self.spectral = SpectralConv2d_FNO(width, modes, modes)
        self.pointwise = nn.Conv3d(width, width, 1)

    def forward(self, x):
        x = self.spectral(x) + self.pointwise(x)
        return F.gelu(x)


class FNO2D(nn.Module):
    """
    Full 2D Fourier Neural Operator.

    Learns a nonlinear operator mapping from spatio-temporal input fields
    to future spatio-temporal output fields.

    Architecture Summary
    --------------------
    1. Appends normalized spatial coordinates as positional encoding

    2. Lifts temporal history into a high-dimensional latent channel space

    3. Applies multiple stacked Fourier Neural Operator blocks

    4. Applies channel-wise MLP to collapse latent channels

    5. Projects intermediate feature representation to output time horizon

    Forward Input
    -------------
    x : torch.Tensor  
        Shape (batch, time_in, nx, ny, features)

    Returns
    -------
    out : torch.Tensor  
        Shape (batch, nx, ny, time_out)
    """

    def __init__(
        self,
        time_in,
        features,
        time_out,
        width=20,
        modes=12,
    ):
        super().__init__()

        self.features = features + 2 

        self.fc0 = nn.Linear(time_in, width)    

        self.block0 = FNOBlock(width, self.features, modes)
        self.block1 = FNOBlock(width, self.features, modes)
        self.block2 = FNOBlock(width, self.features, modes)
        self.block3 = FNOBlock(width, self.features, modes)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc3 = nn.Linear(self.features, 64)
        self.fc4 = nn.Linear(64, time_out)

    def forward(self, x):
        """
        Forward pass of the 2D Fourier Neural Operator.

        1. Appends spatial coordinates (x, y) to input features

        2. Rearranges tensor to expose time dimension for lifting

        3. Applies linear lifting layer (time → latent channels)

        4. Applies stacked FNO blocks
           (spectral convolution + pointwise convolution + skip connections + GELU)

        5. Applies channel-wise MLP to obtain latent feature field

        6. Projects latent features to output time steps

        Parameters
        ----------
        x : torch.Tensor  
            Input tensor of shape (batch, time_in, nx, ny, features)

        Returns
        -------
        out : torch.Tensor  
            Output tensor of shape (batch, nx, ny, time_out)
        """

        b, t, nx, ny, f = x.shape

        grid = self.get_grid(b, t, nx, ny, x.device)
        x = torch.cat((x, grid), dim=-1)   

        x = x.permute(0, 2, 3, 4, 1)          

        x = self.fc0(x)  #lifting time_in dimension to width                    

        x = x.permute(0, 4, 1, 2, 3)           # (b, width, x, y, f+2)

        # stacked FNO blocks
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)


        x = x.permute(0, 2, 3, 4, 1)           # (b, x, y, f+2, width)
        x = F.gelu(self.fc1(x))                # collapse width channels
        x = self.fc2(x).squeeze(-1)            # (b, x, y, f+2)

        x = F.gelu(self.fc3(x))
        x = self.fc4(x)                       # (b, x, y, time_out) project features dimension to required time_out

        return x

    def get_grid(self, b, t, nx, ny, device):
        """
        Generates normalized spatial coordinate grid for positional encoding.

        1. Creates linearly spaced x and y coordinates in [0, 1]

        2. Broadcasts coordinates to match batch and time dimensions

        3. Concatenates x and y coordinates into a 2-channel grid tensor

        Parameters
        ----------
        b : int  
            Batch size.

        t : int  
            Number of time steps.

        nx : int  
            Grid size in x-direction.

        ny : int  
            Grid size in y-direction.

        device : torch.device  
            Device on which the grid is allocated.

        Returns
        -------
        grid : torch.Tensor  
            Tensor of shape (batch, time, nx, ny, 2) containing normalized
            spatial coordinates.
        """

        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)

        gridx = gridx.view(1, 1, nx, 1, 1).repeat(b, t, 1, ny, 1)
        gridy = gridy.view(1, 1, 1, ny, 1).repeat(b, t, nx, 1, 1)

        return torch.cat((gridx, gridy), dim=-1)
