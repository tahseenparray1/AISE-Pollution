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
# MODULE 1: PHYSICS PDE CELL (Updated)
# - Added emission source term
# - CFL-safe sigmoid clamping on all PDE coefficients
# ============================================================

class PhysicsPDECell(nn.Module):
    """
    CFL-safe autoregressive PDE solver for PM2.5 transport.

    Equation per timestep:
        C_{t+1} = C_t + dt * (Advection + Diffusion + Sources - Sinks)

    All coefficients are bounded via sigmoid to prevent numerical divergence.
    """
    # Maximum safe values for CFL stability on a 140x124 grid
    MAX_DT = 0.5
    MAX_DIFFUSION = 0.1
    MAX_SCAV = 0.2
    MAX_EMISSION = 0.1

    def __init__(self):
        super().__init__()

        # Fixed finite-difference filters (non-trainable)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

        # Learnable PDE coefficients (raw logits, clamped via sigmoid * MAX)
        self._dt_logit = nn.Parameter(torch.tensor(0.0))
        self._diff_logit = nn.Parameter(torch.tensor(-1.0))
        self._scav_logit = nn.Parameter(torch.tensor(-1.0))
        self._emission_logit = nn.Parameter(torch.tensor(-1.0))

    @property
    def dt(self):
        return torch.sigmoid(self._dt_logit) * self.MAX_DT

    @property
    def diffusion_coef(self):
        return torch.sigmoid(self._diff_logit) * self.MAX_DIFFUSION

    @property
    def scav_coef(self):
        return torch.sigmoid(self._scav_logit) * self.MAX_SCAV

    @property
    def emission_coef(self):
        return torch.sigmoid(self._emission_logit) * self.MAX_EMISSION

    def _step(self, c, u, v, pblh, rain_mask, emissions):
        """
        Single PDE timestep: C_t → C_{t+1}

        Args:
            c:          (B, 1, H, W)  current PM2.5
            u:          (B, 1, H, W)  u-wind
            v:          (B, 1, H, W)  v-wind
            pblh:       (B, 1, H, W)  boundary layer height
            rain_mask:  (B, 1, H, W)  binary rain indicator
            emissions:  (B, 1, H, W)  total emission proxy (sum of 7 emission channels)
        """
        dc_dx = F.conv2d(c, self.sobel_x, padding=1)
        dc_dy = F.conv2d(c, self.sobel_y, padding=1)
        lapl_c = F.conv2d(c, self.laplacian, padding=1)

        # 1. Advection: -(u·∂C/∂x + v·∂C/∂y)
        advection = -(u * dc_dx + v * dc_dy)

        # 2. Diffusion: K·pblh·∇²C
        diffusion = self.diffusion_coef * (1.0 + F.softplus(pblh)) * lapl_c

        # 3. Sources: emission_coef · emissions (NEW)
        sources = self.emission_coef * emissions

        # 4. Sinks: -Λ·rain·C
        sinks = self.scav_coef * rain_mask * c

        # Forward Euler step (CFL-safe via sigmoid clamping)
        c_next = c + self.dt * (advection + diffusion + sources - sinks)
        return c_next


# ============================================================
# MODULE 3: DYNAMIC PHYSICS GATE (Updated)
# Now receives current weather (u,v,pblh) in addition to static features,
# enabling storm-aware gating decisions.
# ============================================================

class DynamicPhysicsGate(nn.Module):
    """
    Spatiotemporal gate: decides per-pixel whether to trust physics or neural.
    Input: 7(emissions) + 1(topo) + 1(u10) + 1(v10) + 1(pblh) = 11 channels.

    α ≈ 1: trust physics (flat terrain, steady wind)
    α ≈ 0: trust neural (mountains, storms, complex chemistry)
    """
    def __init__(self, in_channels=11):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, static_and_weather):
        return self.gate(static_and_weather)


# ============================================================
# PHYSICS-GATED RECURRENT NEURAL OPERATOR (PGNO-RNN)
# Class name kept as FNO2D for backward-compatible imports.
#
# Architecture: At each of 16 future steps:
#   1. Physics PDE steps C_t → C_physics using wind/rain/pblh/emissions
#   2. WNO neural block computes correction Δ from [C_t, weather, static]
#   3. Dynamic gate blends → C_{t+1} = α·C_physics + (1-α)·(C_t + Δ)
#   4. C_{t+1} feeds into next step
# ============================================================

class FNO2D(nn.Module):
    """
    PDE-RNN: Recurrent Physics-Gated Neural Operator.

    Instead of predicting all 16 hours in one jump, the neural correction
    runs INSIDE the physics autoregressive loop. This ensures sharp, physically
    consistent predictions at all forecast horizons.

    Input layout (B, H, W, 278):
        [0:10]    PM2.5 history (10 hours)
        [10:270]  Temporal (10 features × 26 hours), layout: ch = 10 + t*10 + f
        [270:277] Static emissions (7 channels)
        [277]     Topography (1 channel)
    """

    # Feature indices within the 10 temporal features per timestep
    _F_U10 = 2
    _F_V10 = 3
    _F_PBLH = 5
    _F_RAIN_MASK = 9

    def __init__(self, in_channels, time_out=16, width=128, modes=None, time_input=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input
        self.in_channels = in_channels

        # Dynamically derive num_feats from input channel layout
        # in_channels = time_input(10) + temporal(num_feats*26) + static(7) + topo(1)
        self.num_temporal_feats = (in_channels - time_input - 7 - 1) // 26
        self.total_time = 26  # total temporal window

        # --- Per-Step Neural Correction ---
        # Input per step: C_t(1) + weather(num_feats) + static(8) + grid(2)
        step_in = 1 + self.num_temporal_feats + 8 + 2
        self.step_encoder = nn.Sequential(
            nn.Conv2d(step_in, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.05)
        )

        # 2 WNO blocks (shared across all 16 steps = weight tying regularization)
        self.block0 = WNOBlock(self.width)
        self.block1 = WNOBlock(self.width)

        # Per-step correction decoder → 1 channel (delta)
        self.fc1 = nn.Conv2d(self.width, 64, kernel_size=1)
        self.fc2 = nn.Conv2d(64, 1, kernel_size=1)

        # --- Physics Branch ---
        self.physics_cell = PhysicsPDECell()

        # --- Dynamic Gate (weather-aware) ---
        # Input: emissions(7) + topo(1) + u10(1) + v10(1) + pblh(1) = 11
        self.physics_gate = DynamicPhysicsGate(in_channels=11)

    def get_grid(self, b, nx, ny, device):
        gridx = torch.linspace(0, 1, nx, device=device)
        gridy = torch.linspace(0, 1, ny, device=device)
        gridx = gridx.view(1, 1, nx, 1).repeat(b, 1, 1, ny)
        gridy = gridy.view(1, 1, 1, ny).repeat(b, 1, nx, 1)
        return torch.cat((gridx, gridy), dim=1)

    def forward(self, x):
        """
        PDE-RNN forward pass: autoregressive 16-step loop with
        physics PDE + neural correction + dynamic gating at each step.

        Args:
            x: (B, H, W, 278) full input tensor
        Returns:
            (B, H, W, 16) PM2.5 predictions for 16 future hours
        """
        b, nx, ny, _ = x.shape

        # ========================================
        # EXTRACT ALL FEATURES FROM INPUT
        # ========================================

        # Last known PM2.5 (starting state for autoregressive loop)
        c = x[..., self.time_input - 1:self.time_input].permute(0, 3, 1, 2)  # (B, 1, H, W)

        # Unflatten temporal block → (B, H, W, 26, num_feats)
        temporal_flat = x[..., self.time_input:self.time_input + self.num_temporal_feats * self.total_time]
        temporal = temporal_flat.reshape(b, nx, ny, self.total_time, self.num_temporal_feats)
        # Future hours: temporal[:, :, :, time_input:, :]
        # temporal[:,:,:, 10+h, :] gives weather at future hour h

        # Static features
        emissions = x[..., -8:-1].permute(0, 3, 1, 2)       # (B, 7, H, W)
        topo = x[..., -1:].permute(0, 3, 1, 2)               # (B, 1, H, W)
        static_feats = torch.cat([emissions, topo], dim=1)    # (B, 8, H, W)

        # Total emission proxy (sum of 7 channels → single source map)
        total_emissions = emissions.sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Spatial grid (computed once, reused at every step)
        grid = self.get_grid(b, nx, ny, x.device)             # (B, 2, H, W)

        # ========================================
        # AUTOREGRESSIVE PDE-RNN LOOP
        # ========================================
        outputs = []

        for h in range(self.time_out):
            # --- Extract weather at future hour h ---
            t_idx = self.time_input + h  # absolute hour index (10..25)
            weather_h = temporal[:, :, :, t_idx, :].permute(0, 3, 1, 2)  # (B, num_feats, H, W)

            # Individual physics-relevant features
            u_h = weather_h[:, self._F_U10:self._F_U10+1, :, :]
            v_h = weather_h[:, self._F_V10:self._F_V10+1, :, :]
            pblh_h = weather_h[:, self._F_PBLH:self._F_PBLH+1, :, :]
            rain_h = weather_h[:, self._F_RAIN_MASK:self._F_RAIN_MASK+1, :, :]

            # ---- 1. PHYSICS STEP (with emission sources) ----
            physics_c = self.physics_cell._step(
                c, u_h, v_h, pblh_h, rain_h, total_emissions
            )

            # ---- 2. NEURAL CORRECTION ----
            # Input: [current_pm25, all_weather, static, grid]
            step_input = torch.cat([c, weather_h, static_feats, grid], dim=1)
            feat = self.step_encoder(step_input)
            feat = self.block0(feat)
            feat = self.block1(feat)
            feat = F.gelu(self.fc1(feat))
            neural_delta = self.fc2(feat)                      # (B, 1, H, W)
            neural_c = c + neural_delta                        # residual connection

            # ---- 3. DYNAMIC GATE ----
            gate_input = torch.cat([
                emissions, topo, u_h, v_h, pblh_h
            ], dim=1)                                          # (B, 11, H, W)
            alpha = self.physics_gate(gate_input)              # (B, 1, H, W)

            # ---- 4. BLEND & ADVANCE ----
            c_next = alpha * physics_c + (1.0 - alpha) * neural_c
            outputs.append(c_next)
            c = c_next  # Feed into next step

        # Stack outputs → (B, 16, H, W) → (B, H, W, 16)
        return torch.cat(outputs, dim=1).permute(0, 2, 3, 1)