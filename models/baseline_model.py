import torch
import torch.nn as nn
import torch.nn.functional as F

def check_nan(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"!!! NAN or INF DETECTED AT: {name} !!!")
        return True
    return False

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

class TemporalEncoder(nn.Module):
    """ Extracts temporal derivatives and preserves event timing securely under AMP """
    def __init__(self, in_features, hidden_dim, out_features, total_time=26):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, hidden_dim)  # <-- AMP Safety Net 1
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, hidden_dim)  # <-- AMP Safety Net 2
        
        # Replace pooling with a Flatten + Linear mapping
        self.flatten_dim = hidden_dim * total_time
        self.fc = nn.Linear(self.flatten_dim, out_features)
        self.norm3 = nn.LayerNorm(out_features)   # <-- AMP Safety Net 3

    def forward(self, x):
        # x is expected to be (Batch, Features, Time)
        x1 = self.conv1(x)
        if check_nan(x1, "TemporalEncoder conv1 output"): pass
        x1_n = self.norm1(x1)
        if check_nan(x1_n, "TemporalEncoder norm1 output"): pass
        x1_r = F.relu(x1_n)
        
        x2 = self.conv2(x1_r)
        if check_nan(x2, "TemporalEncoder conv2 output"): pass
        x2_n = self.norm2(x2)
        if check_nan(x2_n, "TemporalEncoder norm2 output"): pass
        x = F.relu(x2_n)
        
        # Flatten the feature and time dimensions
        x = x.view(x.size(0), -1) # Now safe to use .view() here
        x_fc = self.fc(x)            # (Batch, out_features)
        if check_nan(x_fc, "TemporalEncoder fc output"): pass
        x = self.norm3(x_fc)  # Lock the output variance before concatenation
        if check_nan(x, "TemporalEncoder norm3 output"): pass
        
        return x

class WNOBlock(nn.Module):
    """ Wavelet Neural Operator Block with Extended Receptive Field """
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dwt = HaarWavelet2D(width)
        self.idwt = InverseHaarWavelet2D(width)
        
        # 1. Spatial processing per sub-band (Keeps parameter count low)
        self.spectral_spatial = nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1, groups=4)
        
        # ---> AMP OVERFLOW FIX: InstanceNorm2d normalizes per-channel.
        # Sums only 4,340 spatial elements, making it 100% safe from float16 (65,504) overflow.
        self.spectral_norm = nn.InstanceNorm2d(width * 4, affine=True) 
        
        # 2. Cross-frequency mixing (Mixes LL, LH, HL, HH together)
        self.spectral_pointwise = nn.Conv2d(width * 4, width * 4, kernel_size=1, groups=1)
        
        # FIX #4: Depthwise conv with large kernel to expand receptive field
        # Enables modeling long-range wind advection across the 140x124 spatial grid
        self.spatial_mixer = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x):
        if check_nan(x, "WNOBlock input x"): pass
        x_norm = self.norm(x)
        if check_nan(x_norm, "WNOBlock x_norm"): pass
        
        # 1. Decompose to Wavelet domain
        x_w = self.dwt(x_norm)
        if check_nan(x_w, "WNOBlock dwt(x_norm)"): pass
        
        # 2. Mix frequencies (Spatial -> Norm -> Pointwise)
        x_w_sp = self.spectral_spatial(x_w)
        if check_nan(x_w_sp, "WNOBlock spectral_spatial"): pass
        
        x_w_n = self.spectral_norm(x_w_sp)
        if check_nan(x_w_n, "WNOBlock spectral_norm"): pass
        
        x_w_pw = self.spectral_pointwise(x_w_n)
        if check_nan(x_w_pw, "WNOBlock spectral_pointwise"): pass
        
        # 3. Reconstruct back to Spatial domain
        out_w = self.idwt(x_w_pw)
        if check_nan(out_w, "WNOBlock idwt"): pass
        
        # 4. Apply large-kernel spatial mixing for extended receptive field
        out_w_m = self.spatial_mixer(out_w)
        if check_nan(out_w_m, "WNOBlock spatial_mixer"): pass
        
        # 5. Add pointwise mixing and residual
        pw_mix = self.pointwise(x_norm)
        if check_nan(pw_mix, "WNOBlock pointwise mix"): pass
        
        out = out_w_m + pw_mix
        return x + F.gelu(out)

class FNO2D(nn.Module):
    # Renamed internal structure to WNO, kept class name FNO2D to prevent breaking your train.py imports!
    def __init__(self, in_channels, time_out=16, width=64, modes=None, time_input=10, total_time=26, num_temporal_features=10):
        super().__init__()
        self.width = width
        self.time_out = time_out
        self.time_input = time_input  # Number of PM2.5 history hours (for residual connection)
        self.total_time = total_time
        self.num_temporal_features = num_temporal_features
        
        self.temporal_encoder = TemporalEncoder(num_temporal_features, 32, 32, total_time=total_time)
        
        # New in channels calculation: (pm history) + (encoded temporal) + (static & topo)
        original_temporal_channels = self.total_time * self.num_temporal_features
        static_topo_channels = in_channels - self.time_input - original_temporal_channels
        
        new_in_channels = self.time_input + 32 + static_topo_channels
        
        # Encode ablated input down to 'width'
        # FIX #3: Use standard Dropout instead of Dropout2d to avoid
        # dropping entire temporal channels (e.g. "Hour 5 Wind Speed")
        self.input_encoder = nn.Sequential(
            nn.Conv2d(new_in_channels + 2, width, kernel_size=1),
            nn.GroupNorm(4, width),
            nn.GELU(),
            nn.Dropout(p=0.20)
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
        if check_nan(x, "FNO2D VERY FIRST INPUT"): pass
        
        b, nx, ny, _ = x.shape
        # Extract last known PM2.5 state for residual connection
        last_pm25 = x[..., self.time_input-1:self.time_input].permute(0, 3, 1, 2) 
        
        # Slice X into components
        c0 = self.time_input
        c1 = c0 + self.total_time * self.num_temporal_features
        
        pm = x[..., :c0]
        temporal = x[..., c0:c1]
        static_topo = x[..., c1:]
        
        # Process temporal through TemporalEncoder
        # temporal shape: (b, nx, ny, total_time * num_temporal_features)
        assert temporal.shape[-1] == self.total_time * self.num_temporal_features, "Incoming temporal channels do not match config!"
        
        temporal = temporal.reshape(b, nx, ny, self.total_time, self.num_temporal_features)
        # (b * nx * ny, num_temporal_features, total_time) for Conv1d
        temporal_flat = temporal.permute(0, 1, 2, 4, 3).reshape(-1, self.num_temporal_features, self.total_time)
        
        temporal_encoded = self.temporal_encoder(temporal_flat) # (b * nx * ny, 32)
        if check_nan(temporal_encoded, "FNO2D AFTER temporal_encoder"): pass
        temporal_encoded = temporal_encoded.reshape(b, nx, ny, 32)
        
        # Re-concatenate
        x_new = torch.cat([pm, temporal_encoded, static_topo], dim=-1)
        
        x_in = x_new.permute(0, 3, 1, 2)
        grid = self.get_grid(b, nx, ny, x.device) 
        x_in = torch.cat([x_in, grid], dim=1) 
        
        x_feat = self.input_encoder(x_in)
        if check_nan(x_feat, "FNO2D AFTER input_encoder"): pass
        
        # WNO Spatial Mixing (No padding needed, 140 and 124 divide evenly by 2!)
        x_wno = self.block0(x_feat)
        if check_nan(x_wno, "FNO2D AFTER block0"): pass
        
        x_wno = self.block1(x_wno)
        if check_nan(x_wno, "FNO2D AFTER block1"): pass
        
        x_wno = self.block2(x_wno)
        if check_nan(x_wno, "FNO2D AFTER block2"): pass
        
        x_wno = self.block3(x_wno)
        if check_nan(x_wno, "FNO2D AFTER block3"): pass
        
        # Decode to DELTA (network learns the change from current state)
        x_wno = self.fc1(x_wno)
        if check_nan(x_wno, "FNO2D AFTER fc1"): pass
        
        x_wno = F.gelu(x_wno)
        out = self.fc2(x_wno) 
        if check_nan(out, "FNO2D AFTER fc2"): pass
        
        # FIX #1: RESIDUAL CONNECTION
        # 'out' shape is (B, 16, H, W). 'last_pm25' shape is (B, 1, H, W).
        # Broadcasting adds the last known PM2.5 state to all 16 future steps.
        # This lets the network learn DELTA (change) instead of absolute values.
        out = out + last_pm25
        
        return out.permute(0, 2, 3, 1)