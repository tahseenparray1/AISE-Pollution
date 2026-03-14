import torch
import torch.nn as nn
from models.baseline_model import WNO_WUNet

def trace_nan(module, input, output):
    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                print(f"!!! NAN or INF DETECTED AT output {i} of {module.__class__.__name__} !!!")
    else:
        if isinstance(output, torch.Tensor) and (torch.isnan(output).any() or torch.isinf(output).any()):
            print(f"!!! NAN or INF DETECTED AT output of {module.__class__.__name__} !!!")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Using the sizes from the model execution
    num_temporal_features = 10
    total_time = 26
    in_channels = 292
    
    model = WNO_WUNet(
        in_channels=in_channels,
        out_channels=16,
        modes=8,
        width=256,
        time_input=10,
        total_time=total_time,
        num_temporal_features=num_temporal_features
    ).to(device)
    
    # We already have check_nan in baseline_model manually added!
    
    # Register hooks just in case
    for name, layer in model.named_modules():
        layer.register_forward_hook(trace_nan)

    # Batch 2, H=140, W=124, C=292
    x = torch.randn(2, 140, 124, in_channels, device=device)
    # Inject an outlier resembling a 2500 PM2.5 normalized value (e.g. 100.0)
    x[0, 70, 60, 15] = 100.0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    use_amp = (device.type == 'cuda')
    # If on CPU, autocast still works for bfloat16, but we really want to test float16.
    # We will force autocast with float16
    scaler = torch.amp.GradScaler(device.type, enabled=True) if device.type == 'cuda' else torch.amp.GradScaler('cpu', enabled=False)
    
    print("\n--- Running Forward Pass (AMP) ---")
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True): # Force float16 testing
            out = model(x)
        
        print(f"Output shape: {out.shape}")
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("ERROR: FINAL OUTPUT CONTAINS NAN/INF")
        else:
            print("Forward pass successful. No NaNs!")
            
        print("\n--- Running Backward Pass ---")
        out_f32 = out.float()
        loss = out_f32.sum()
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            print("Backward pass successful!")
        else:
            loss.backward()
            print("Backward pass successful!")
            
    except Exception as e:
        print(f"Execution Failed: {e}")

if __name__ == '__main__':
    main()
