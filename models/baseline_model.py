import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# WAVELET TRANSFORM PRIMITIVES (Unchanged)
# ============================================================

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

# ============================================================
# WAVELET NEURAL OPERATOR BLOCK (Unchanged)
# ============================================================

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

# ============================================================
# MODULE 1: PHYSICS PDE CELL
# Implements the 2D Advection-Diffusion-Reaction equation
# using fixed (non-trainable) Sobel & Laplacian convolution
# filters with learnable PDE coefficients.
# ============================================================

class PhysicsPDECell(nn.Module):
    """
    Autoregressive PDE solver for PM2.5 transport.
    
    Equation per timestep:
        C_{t+1} = C_t + dt * (Advection + Diffusion + Sinks)
    
    Where:
        Advection = -(u * dC/dx + v * dC/dy)        [wind transport]
        Diffusion = K * pblh * (d²C/dx² + d²C/dy²)  [turbulent mixing]
        Sink      = -Λ * rain_mask * C_t             [wet scavenging]
    """
    def __init__(self):
        super().__init__()
        
        # ---- Fixed Finite-Difference Filters (Non-trainable) ----
        
        # Sobel-X: approximates ∂C/∂x (central difference)
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32) / 8.0
        
        # Sobel-Y: approximates ∂C/∂y (central difference)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32) / 8.0
        
        # Laplacian: approximates ∂²C/∂x² + ∂²C/∂y² (isotropic)
        laplacian = torch.tensor([[ 0.,  1.,  0.],
                                  [ 1., -4.,  1.],
                                  [ 0.,  1.,  0.]], dtype=torch.float32)
        
        # Register as non-trainable buffers, shaped for F.conv2d: (1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        
        # ---- Learnable PDE Coefficients ----
        # These scalars absorb unit conversion between normalized features and physical units.
        # Initialized to small values for stable early training.
        self.dt = nn.Parameter(torch.tensor(0.1))               # Timestep scale
        self.diffusion_coef = nn.Parameter(torch.tensor(0.01))  # K (diffusivity)
        self.scav_coef = nn.Parameter(torch.tensor(0.05))       # Λ (rain washout rate)
    
    def _step(self, c, u, v, pblh, rain_mask):
        """
        Single PDE timestep: C_{t} → C_{t+1}
        
        Args:
            c:         (B, 1, H, W) current PM2.5 concentration (normalized)
            u:         (B, 1, H, W) zonal wind u10 (normalized)
            v:         (B, 1, H, W) meridional wind v10 (normalized)
            pblh:      (B, 1, H, W) planetary boundary layer height (normalized)
            rain_mask: (B, 1, H, W) binary rain indicator (0 or ~1)
        
        Returns:
            c_next:    (B, 1, H, W) next timestep PM2.5
        """
        # Compute spatial gradients of concentration
        dc_dx = F.conv2d(c, self.sobel_x, padding=1)   # (B, 1, H, W)
        dc_dy = F.conv2d(c, self.sobel_y, padding=1)   # (B, 1, H, W)
        lapl_c = F.conv2d(c, self.laplacian, padding=1) # (B, 1, H, W)
        
        # 1. Advection: -( u * ∂C/∂x + v * ∂C/∂y )
        advection = -(u * dc_dx + v * dc_dy)
        
        # 2. Diffusion: K * pblh * ∇²C  (pblh modulates turbulent mixing)
        diffusion = self.diffusion_coef * (1.0 + F.softplus(pblh)) * lapl_c
        
        # 3. Wet Scavenging: -Λ * rain * C
        scavenging = -self.scav_coef * rain_mask * c
        
        # Forward Euler step
        c_next = c + self.dt * (advection + diffusion + scavenging)
        
        return c_next
    
    def forward(self, last_pm25, future_u10, future_v10, future_pblh, future_rain_mask):
        """
        Autoregressive 16-step PDE rollout.
        
        Args:
            last_pm25:        (B, 1, H, W)  PM2.5 at the last known hour
            future_u10:       (B, 16, H, W) u-wind at future hours 0..15
            future_v10:       (B, 16, H, W) v-wind at future hours 0..15
            future_pblh:      (B, 16, H, W) PBLH at future hours 0..15
            future_rain_mask: (B, 16, H, W) rain indicator at future hours 0..15
        
        Returns:
            predictions: (B, 16, H, W) physics-predicted PM2.5 for 16 future hours
        """
        c = last_pm25  # (B, 1, H, W) - starting state
        outputs = []
        
        for h in range(16):
            u = future_u10[:, h:h+1, :, :]       # (B, 1, H, W)
            v = future_v10[:, h:h+1, :, :]       # (B, 1, H, W)
            pblh = future_pblh[:, h:h+1, :, :]   # (B, 1, H, W)
            rain = future_rain_mask[:, h:h+1, :, :] # (B, 1, H, W)
            
            c = self._step(c, u, v, pblh, rain)
            outputs.append(c)
        
        return torch.cat(outputs, dim=1)  # (B, 16, H, W)

# ============================================================
# MODULE 3: SPATIALLY-AWARE PHYSICS GATE (SAPG)
# Outputs α ∈ [0,1] per pixel: 1 = trust physics, 0 = trust WNO
# ============================================================

class SpatialPhysicsGate(nn.Module):
    """
    Lightweight convolutional gate that decides how much to trust
    the physics branch vs the neural branch at each spatial location.
    
    Takes static features (topography + emissions) that characterize
    the local geographic complexity. Over flat plains (Indo-Gangetic),
    α → 1. Over complex terrain (Himalayas), α → 0.
    """
    def __init__(self, in_channels=8):
        super().__init__()
        # in_channels = 7 (emissions) + 1 (topo) = 8
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, static_features):
        """
        Args:
            static_features: (B, 8, H, W) — 7 emission channels + 1 topo
        
        Returns:
            alpha: (B, 1, H, W) — physics trust map, values in [0, 1]
        """
        return self.gate(static_features)

# ============================================================
# PHYSICS-GATED NEURAL OPERATOR (PGNO)
# Class name kept as FNO2D for backward-compatible imports.
# ============================================================

class FNO2D(nn.Module):
    """
    Physics-Gated Neural Operator (PGNO).
    
    Runs a WNO neural backbone and a Physics PDE solver in parallel,
    then blends them pixel-by-pixel using a learned spatial gate:
    
        out = α · physics_abs + (1 - α) · neural_abs
    
    Channel layout of input (B, H, W, 278):
        [ 0:10  ] PM2.5 history (10 hours)
        [ 10:270] Temporal features (10 features × 26 hours)
                   Layout: channel = 10 + t*10 + f
                   f=0:q2, f=1:t2, f=2:u10, f=3:v10, f=4:swdown,
                   f=5:pblh, f=6:rain, f=7:wind_speed, f=8:vent_coef, f=9:rain_mask
        [270:277] Static emissions (7 channels)
        [277    ] Topography (1 channel)
    """
    
    # Feature indices within the temporal block (10 features per timestep)
    _F_U10 = 2
    _F_V10 = 3
    _F_PBLH = 5
    _F_RAIN_MASK = 9
    
    def __init__(self, in_channels, time_out=16, width=64, modes=None, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input
        self.in_channels = in_channels
        
        # --- Neural Branch (WNO Backbone) ---
        self.input_encoder = nn.Sequential(
            nn.Conv2d(in_channels + 2, width, kernel_size=1),  # +2 for spatial grid
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.05)
        )
        
        self.block0 = WNOBlock(self.width)
        self.block1 = WNOBlock(self.width)
        self.block2 = WNOBlock(self.width)
        self.block3 = WNOBlock(self.width)
        
        self.fc1 = nn.Conv2d(self.width, 128, kernel_size=1)
        self.fc2 = nn.Conv2d(128, self.time_out, kernel_size=1)
        
        # --- Physics Branch ---
        self.physics_cell = PhysicsPDECell()
        
        # --- Spatial Gate ---
        # Input: 7 emissions + 1 topo = 8 static channels
        self.physics_gate = SpatialPhysicsGate(in_channels=8)
    
    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1)
    
    def _extract_future_feature(self, x, feature_idx):
        """
        Extract a specific meteorological feature at all 16 future hours
        from the flattened 278-channel input tensor.
        
        Args:
            x: (B, H, W, 278) input tensor
            feature_idx: index within the 10 temporal features (e.g., 2 for u10)
        
        Returns:
            (B, 16, H, W) tensor for that feature at future hours 0..15
        """
        channels = []
        pm_offset = self.time_input  # 10 (PM2.5 history channels)
        num_feats = 10               # 10 temporal features per hour
        
        for h in range(self.time_out):  # 16 future steps
            t = self.time_input + h     # absolute hour index (10..25)
            ch_idx = pm_offset + t * num_feats + feature_idx
            channels.append(x[..., ch_idx])  # (B, H, W)
        
        return torch.stack(channels, dim=1)  # (B, 16, H, W)
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, 278) full input tensor
        
        Returns:
            (B, H, W, 16) blended PM2.5 predictions for 16 future hours
        """
        b, nx, ny, _ = x.shape
        
        # ============================================
        # 1. EXTRACT PHYSICS-RELEVANT CHANNELS
        # ============================================
        
        # Last known PM2.5 for residual connection
        last_pm25 = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)  # (B, 1, H, W)
        
        # Future meteorology for PDE cell (16 future hours each)
        future_u10 = self._extract_future_feature(x, self._F_U10)          # (B, 16, H, W)
        future_v10 = self._extract_future_feature(x, self._F_V10)          # (B, 16, H, W)
        future_pblh = self._extract_future_feature(x, self._F_PBLH)        # (B, 16, H, W)
        future_rain_mask = self._extract_future_feature(x, self._F_RAIN_MASK)  # (B, 16, H, W)
        
        # Static features for the gate:  emissions (7ch) + topo (1ch)
        emissions = x[..., -8:-1].permute(0, 3, 1, 2)  # (B, 7, H, W) channels 270-276
        topo = x[..., -1:].permute(0, 3, 1, 2)          # (B, 1, H, W) channel 277
        gate_input = torch.cat([emissions, topo], dim=1)  # (B, 8, H, W)
        
        # ============================================
        # 2. NEURAL BRANCH (WNO)
        # ============================================
        x_in = x.permute(0, 3, 1, 2)                       # (B, 278, H, W)
        grid = self.get_grid(b, nx, ny, x.device)           # (B, 2, H, W)
        x_in = torch.cat([x_in, grid], dim=1)               # (B, 280, H, W)
        
        x_feat = self.input_encoder(x_in)                   # (B, width, H, W)
        
        x_wno = self.block0(x_feat)
        x_wno = self.block1(x_wno)
        x_wno = self.block2(x_wno)
        x_wno = self.block3(x_wno)
        
        x_wno = F.gelu(self.fc1(x_wno))                     # (B, 128, H, W)
        neural_delta = self.fc2(x_wno)                       # (B, 16, H, W)
        
        # Neural absolute prediction = delta + last known state
        neural_abs = neural_delta + last_pm25                 # (B, 16, H, W)
        
        # ============================================
        # 3. PHYSICS BRANCH (PDE Cell)
        # ============================================
        physics_abs = self.physics_cell(
            last_pm25, future_u10, future_v10, future_pblh, future_rain_mask
        )  # (B, 16, H, W)
        
        # ============================================
        # 4. SPATIAL GATE + BLENDING
        # ============================================
        alpha = self.physics_gate(gate_input)                 # (B, 1, H, W)
        
        # Blend: α * physics + (1-α) * neural,  per pixel
        out = alpha * physics_abs + (1.0 - alpha) * neural_abs  # (B, 16, H, W)
        
        return out.permute(0, 2, 3, 1)  # (B, H, W, 16)